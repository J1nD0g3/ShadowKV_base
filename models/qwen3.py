################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


import torch
import torch.nn.functional as F
import gc
import time

import transformers
from transformers import Qwen3ForCausalLM, Qwen3Config, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
transformers.logging.set_verbosity_error()

from .tensor_op import layer_norm, apply_rotary_pos_emb, apply_rotary_pos_emb_single, sample_token
from .prompt_template import Templates, Chat_Templates
from .base import LLM

class Qwen3Layer:
    def __init__(self, layer_idx) -> None:

        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_proj :torch.Tensor = None
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        # Qwen3-specific: QK normalization
        self.q_norm_weight :torch.Tensor = None
        self.q_norm_variance_epsilon :float = 0.0
        self.k_norm_weight :torch.Tensor = None
        self.k_norm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx

    def init_parameters(self, hf_layer: Qwen3DecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()

        # Qwen3: no QKV bias (attention_bias=False)

        # Qwen3-specific: QK normalization weights
        self.q_norm_weight = hf_layer.self_attn.q_norm.weight.detach()
        self.q_norm_variance_epsilon = hf_layer.self_attn.q_norm.variance_epsilon
        self.k_norm_weight = hf_layer.self_attn.k_norm.weight.detach()
        self.k_norm_variance_epsilon = hf_layer.self_attn.k_norm.variance_epsilon

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device, non_blocking=True)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device, non_blocking=True)
        self.wq = self.wq.to(device, non_blocking=True)
        self.wk = self.wk.to(device, non_blocking=True)
        self.wv = self.wv.to(device, non_blocking=True)
        self.wo = self.wo.to(device, non_blocking=True)
        self.gate_proj = self.gate_proj.to(device, non_blocking=True)
        self.up_proj = self.up_proj.to(device, non_blocking=True)
        self.down_proj =  self.down_proj.to(device, non_blocking=True)

        self.q_norm_weight = self.q_norm_weight.to(device, non_blocking=True)
        self.k_norm_weight = self.k_norm_weight.to(device, non_blocking=True)

class Qwen3(LLM):
    def __init__(self,
        model_name: str = "Qwen/Qwen3-8B",
        batch_size :int = 1,
        max_length :int = 64*1024,
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        attn_mode: str = 'full',
        sparse_budget: int = 2048,
        rank=160,
        chunk_size=8,
        minference=False,
        enable_thinking=False) -> None:

        assert batch_size == 1, "Batch size must be 1"
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.config = Qwen3Config.from_pretrained(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.enable_thinking = enable_thinking

        self.init_parameters()
        self.attn_mode = attn_mode
        self.minference = minference

        if self.enable_thinking:
            self.ctx_template = Templates['qwen3_thinking']
            self.chat_template = Chat_Templates['qwen3_thinking']
        else:
            self.ctx_template = Templates['qwen3']
            self.chat_template = Chat_Templates['qwen3']

        self.init_kv_cache(sparse_budget, rank, chunk_size, self.config)

    def _set_cos_sin_cache(self, inv_freq: torch.Tensor):
        t = torch.arange(self.max_length, device=self.device, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_parameters(self):
        hf_model = Qwen3ForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)
        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.to(self.device))
        # cos_sin_cache for ShadowKV decode path (base.py layer_compute)
        self.cos_sin_cache = torch.cat((self.cos_cache[:, :self.head_dim // 2], self.sin_cache[:, :self.head_dim // 2]), dim=-1)
        self.layers :list[Qwen3Layer] = []

        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = Qwen3Layer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            hf_model.model.layers[idx] = None
            gc.collect()

        self.num_layers = len(self.layers)

    def _qk_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply RMSNorm on Q or K per head (last dim = head_dim)."""
        return layer_norm(x.contiguous(), eps, weight)

    def pre_attention_compute(
        self,
        hidden_states: torch.Tensor,
        buffer: Qwen3Layer,
        num_heads:int,
        num_key_value_heads:int,
        head_dim:int
    ):
        hidden_states = layer_norm(hidden_states, buffer.input_layernorm_variance_epsilon, buffer.input_layernorm_weight)
        bsz, q_len, _ = hidden_states.size()
        # Qwen3: no bias in QKV projections
        query_states = F.linear(hidden_states, buffer.wq)
        key_states = F.linear(hidden_states, buffer.wk)
        value_states = F.linear(hidden_states, buffer.wv)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        # Qwen3-specific: apply QK normalization (RMSNorm per head)
        query_states = self._qk_norm(query_states, buffer.q_norm_weight, buffer.q_norm_variance_epsilon)
        key_states = self._qk_norm(key_states, buffer.k_norm_weight, buffer.k_norm_variance_epsilon)

        return query_states, key_states, value_states

    def post_attention_compute(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
        buffer: Qwen3Layer
    ):
        hidden_states = F.linear(attn_output, buffer.wo)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm(hidden_states, buffer.post_attention_layernorm_variance_epsilon, buffer.post_attention_layernorm_weight)
        up = F.linear(hidden_states, buffer.up_proj)
        gate = F.silu(F.linear(hidden_states, buffer.gate_proj))
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj)
        hidden_states = residual + hidden_states
        return hidden_states

    @torch.inference_mode()
    def apply_rotary_pos_emb_single(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb_single(x, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(q, k, self.cos_cache, self.sin_cache, position_ids)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k: int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        """Qwen3 generate with thinking token handling."""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)

        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())

        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()

        # Qwen3 special token IDs
        think_token_id = self.tokenizer.encode("<think>", add_special_tokens=False)[0]      # 151667
        end_think_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]  # 151668
        im_end_token_id = self.tokenizer.eos_token_id  # <|im_end|> = 151645

        in_thinking = False
        if not self.enable_thinking:
            # When thinking is disabled, we should not enter thinking mode
            pass

        while n < gen_len:
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)

            n += 1
            generated_ids.extend(next_token[0].tolist())

            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=not self.enable_thinking,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            token_id = next_token[0].item()

            # Track thinking state
            if token_id == think_token_id:
                in_thinking = True
            elif token_id == end_think_token_id:
                in_thinking = False

            # Stop conditions
            if token_id == im_end_token_id:
                break
            if token_id == self.tokenizer.convert_tokens_to_ids("<|endoftext|>"):
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        raw_gen_len = len(generated_ids)
        # Strip thinking content: extract text after </think> if present
        if '</think>' in raw_text:
            raw_text = raw_text.split('</think>')[-1].strip()
        elif '<think>' in raw_text:
            # Thinking started but never ended (gen_len too short) — no answer produced
            raw_text = ''
        # Remove stop tokens from output
        for stop_tok in ['<|im_end|>', '<|endoftext|>']:
            raw_text = raw_text.replace(stop_tok, '')
        self._last_raw_gen_len = raw_gen_len
        return [raw_text.strip()]
