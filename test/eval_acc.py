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

# OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 8 test/eval_acc.py --datalen 131072 --method shadowKV --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2" --sparse_budget 896 --rank 160 --chunk_size 8

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime

class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process

def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=60*90))
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{rank}" 
        torch.cuda.set_device(device)
        master_process = (
            rank == 0
        )
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", 'cyan'))
    
    return DistConfig(is_distributed, rank, world_size, device, master_process)

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k")
    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--minference", action='store_true', default=False)
    p.add_argument("--enable_thinking", action='store_true', default=False, help="Enable Qwen3 thinking mode")
    p.add_argument("--sparse_budget_ratio", type=float, default=None,
                   help="Set sparse_budget as a ratio of input length (e.g., 0.02 = 2%%). "
                        "Overrides --sparse_budget. Budget is rounded to chunk_size multiple.")
    p.add_argument("--disable_chat_template", action='store_true', default=False,
                   help="Skip chat template application (useful for code completion tasks)")

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference

    dist_config = init_dist()
    
    from evaluator import Evaluator
    from models import choose_model_class
    from data.dataset import Dataset
    
    evaluator = Evaluator(dist_config, args=args)
    
    if dist_config.master_process:
        print(colored(f"data_names: {dataset_names}", 'cyan'))
    
    LLM = choose_model_class(model_name)

    extra_kwargs = {}
    if 'qwen3' in model_name.lower():
        extra_kwargs['enable_thinking'] = args.enable_thinking
    # max_length: datalen * 2 for thinking mode (input + generation), datalen + 2048 otherwise
    max_length = datalen * 2 if args.enable_thinking else datalen + 2048
    llm = LLM(model_name=model_name, batch_size=batch_size, device=dist_config.device, max_length=max_length, attn_mode=args.method, dtype=dtype, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, minference=minference, **extra_kwargs)

    if dist_config.master_process:
        llm.print_kv_stats()

    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name, llm.tokenizer, datalen, num_samples, evaluator.dist_config.rank, evaluator.dist_config.world_size, enable_thinking=args.enable_thinking, disable_chat_template=args.disable_chat_template)
        budget_label = f"ratio{args.sparse_budget_ratio}" if args.sparse_budget_ratio else str(sparse_budget)
        evaluator.test(llm, dataset, f"archive/{model_name.split('/')[-1]}/{dataset_name}_{datalen}_{args.method}_{budget_label}_{rank}_{chunk_size}.jsonl", args.method, sparse_budget_ratio=args.sparse_budget_ratio)
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    evaluator.summarize()