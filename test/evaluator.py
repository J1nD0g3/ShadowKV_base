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

import os
import sys
import io
import time
import math
import torch
from termcolor import colored
from tqdm import tqdm
import torch.distributed as dist
import pandas as pd
import json
import datetime

from data.dataset import Dataset
from data.metrics import math_verify_score
from models.base import LLM


class TeeWriter:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, 'w', encoding='utf8')

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()

    @property
    def encoding(self):
        return getattr(self.stream, 'encoding', 'utf-8')


class Evaluator:
    def __init__(self, dist_config, args=None):

        self.dist_config = dist_config
        self.args = args

        # init final report
        self.all_stats = []
        self.run_start_time = time.time()

        # Build run directory: logs/Model_dataset_think_timestamp/
        if args and dist_config.master_process:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = args.model_name.split('/')[-1]
            # Extract top-level dataset name (e.g., "longbench" from "longbench/hotpotqa")
            ds_names = args.dataset_name if isinstance(args.dataset_name, list) else [args.dataset_name]
            ds_tops = sorted(set(d.split('/')[0] for d in ds_names))
            datasets_label = '_'.join(ds_tops)
            think_label = 'think_on' if args.enable_thinking else 'think_off'
            self.run_dir = os.path.join("logs", f"{model_short}_{datasets_label}_{think_label}_{timestamp}")
            self.samples_dir = os.path.join(self.run_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)

            # Start progress log tee
            self.progress_path = os.path.join(self.run_dir, "progress.log")
            self._tee_stderr = TeeWriter(sys.stderr, self.progress_path)
            sys.stderr = self._tee_stderr
        else:
            self.run_dir = None
            self.samples_dir = None

    def test(self, llm: LLM, dataset: Dataset, output_path: str, setting: str = 'baseline', sparse_budget_ratio: float = None):

        # mkdir if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.dist_config.master_process:
            print(colored(f"[Test] {llm.model_name} on {dataset.dataset_name}, results saved to {output_path}", 'green'))

        if dataset.is_sharded == False:
            dataset.shard(self.dist_config.rank, self.dist_config.world_size)

        bsz = llm.batch_size
        scores = []
        preds = []
        math_verify_scores = []  # For MATH500 math_verify metric
        sample_details = []      # Per-sample details
        task_start_time = time.time()

        # clear the file
        open(output_path, 'w').close()
        if self.dist_config.is_distributed:
            dist.barrier()

        progress_bar = tqdm(range(dataset.num_samples // bsz), desc=f'Running {dataset.dataset_name}', disable=self.dist_config.is_distributed and not self.dist_config.master_process, ascii=' ░▒█', ncols=120)
        for i in range(dataset.num_samples // bsz):
            prompt = torch.cat([dataset.tokenized_prompts[i*bsz+j] for j in range(bsz)], dim=0)
            input_len = prompt.size(1)
            # Dynamic gen_len: use remaining capacity up to dataset default
            gen_len = min(dataset.gen_len, llm.max_length - input_len)
            gen_len = max(gen_len, 1)  # at least 1 token

            if sparse_budget_ratio is not None:
                total_budget = int(input_len * sparse_budget_ratio)
                # Subtract outlier and local tokens so the total KV matches the ratio
                kv = llm.kv_cache
                outlier_tokens = getattr(kv, 'outlier_chunk', 0) * getattr(kv, 'chunk_size', 1)
                local_tokens = getattr(kv, 'local_chunk', 0) * getattr(kv, 'chunk_size', 1)
                new_budget = max(total_budget - outlier_tokens - local_tokens, getattr(kv, 'chunk_size', 8))
                llm.reinit_kv_cache_with_budget(new_budget)

            if 'persona' in dataset.dataset_name:
                assert bsz == 1
                llm.generate(prompt.to(llm.device), gen_len=0, verbose=False, top_p=0.9, temperature=0.0) # prefill ctx
                queries = dataset.queries[i]
                gts_list = dataset.gt[i]
                rets_list = []
                for query, gts in zip(queries, gts_list):
                    rets = llm.generate(llm.encode(query, template="chat"), cont=True, gen_len=gen_len, top_p=0.9, temperature=0.0)
                    rets_list.append(rets)
                    pred = rets[0]
                    if isinstance(gts, list):
                        if len(gts) == 1:
                            gts = gts[0]
                    scores.append(dataset.metric(pred, gts))

            elif 'long_bench' in dataset.dataset_name:
                rets = llm.generate(prompt.to(llm.device), gen_len=gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for (pred, gt, classes) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz], dataset.classes[i*bsz:(i+1)*bsz]):
                    scores.append(max([dataset.metric(pred, g, classes) for g in gt]))

            elif dataset.dataset_name.startswith('longbench/'):
                rets = llm.generate(prompt.to(llm.device), gen_len=gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for (pred, gt, classes) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz], dataset.all_classes[i*bsz:(i+1)*bsz]):
                    scores.append(dataset.metric(pred, gt, all_classes=classes))

            elif dataset.dataset_name == 'math500':
                rets = llm.generate(prompt.to(llm.device), gen_len=gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for j, (pred, gt) in enumerate(zip(rets, dataset.gt[i*bsz:(i+1)*bsz])):
                    # Truncate at "Problem:" stop word (same as minerva_math eval)
                    if 'Problem:' in pred:
                        pred = pred[:pred.index('Problem:')].strip()
                        rets[j] = pred
                    scores.append(dataset.metric(pred, gt))
                    math_verify_scores.append(math_verify_score(pred, gt))

            else:
                rets = llm.generate(prompt.to(llm.device), gen_len=gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for (pred, gt) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            gt = gt[0]
                    scores.append(dataset.metric(pred, gt))

            # Capture KV budget info after generate (select_sets set during prefill)
            kv_info = {}
            if hasattr(llm, 'kv_cache'):
                kv = llm.kv_cache
                if hasattr(kv, 'sparse_budget'):
                    outlier_tokens = getattr(kv, 'outlier_chunk', 0) * getattr(kv, 'chunk_size', 1)
                    local_tokens = getattr(kv, 'local_chunk', 0) * getattr(kv, 'chunk_size', 1)
                    kv_info['sparse_budget'] = kv.sparse_budget
                    kv_info['total_kv_budget'] = kv.sparse_budget + outlier_tokens + local_tokens
                    kv_info['outlier_chunk'] = getattr(kv, 'outlier_chunk', 0)
                    kv_info['local_chunk'] = getattr(kv, 'local_chunk', 0)
                    kv_info['select_sets'] = getattr(kv, 'select_sets', None)
                    kv_info['chunk_size'] = getattr(kv, 'chunk_size', None)

            # Compute output length from predictions
            if 'persona' in dataset.dataset_name:
                output_len = sum(len(llm.tokenizer.encode(r[0], add_special_tokens=False)) for r in rets_list)
            else:
                output_len = sum(len(llm.tokenizer.encode(r, add_special_tokens=False)) for r in rets)

            # Raw generation length (including thinking tokens, before strip)
            raw_gen_len = getattr(llm, '_last_raw_gen_len', None)

            detail = {
                'input_len': input_len,
                'output_len': output_len,
                'raw_gen_len': raw_gen_len,
            }
            detail.update(kv_info)
            sample_details.append(detail)

            avg_score = sum(scores) / len(scores)
            postfix_str = f"avg={avg_score:.2f}"
            if dataset.dataset_name == 'math500' and math_verify_scores:
                mv = sum(math_verify_scores) / len(math_verify_scores)
                postfix_str += f", mv={mv:.2f}"
            progress_bar.set_postfix_str(postfix_str, refresh=False)
            progress_bar.update(1)

            if dataset.dataset_name == 'niah':
                preds = {
                        "context_length": dataset.ctx_len[i*bsz:(i+1)*bsz],
                        "depth_percent": dataset.depth_pct[i*bsz:(i+1)*bsz],
                        "response": rets,
                        "answer": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores[i*bsz:(i+1)*bsz],
                        "avg_score": avg_score,
                    }
            elif 'persona' in dataset.dataset_name:
                preds = {
                        "query": queries,
                        "response": rets_list,
                        "answer": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores,
                        "avg_score": avg_score,
                    }
            elif dataset.dataset_name == 'math500':
                avg_mv = sum(math_verify_scores) / len(math_verify_scores) if math_verify_scores else 0.0
                preds = {
                        "prediction": rets,
                        "ground_truth": dataset.gt[i*bsz:(i+1)*bsz],
                        "exact_match": scores,
                        "math_verify": math_verify_scores,
                        "avg_exact_match": avg_score,
                        "avg_math_verify": avg_mv,
                    }
            else:
                preds = {
                        "prediction": rets,
                        "ground_truth": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores,
                        "avg_score": avg_score,
                    }

            # Add per-sample detail to jsonl output
            preds['input_len'] = input_len
            preds['output_len'] = output_len
            preds.update(kv_info)

            with open(output_path, "a", encoding="utf8") as fout:
                fout.write(json.dumps(preds, ensure_ascii=False) + "\n")

        progress_bar.close()
        task_elapsed = time.time() - task_start_time
        avg_score = sum(scores) / len(scores)

        # GPU memory stats
        gpu_mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GiB
        gpu_mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)    # GiB

        stat = {
            'model': llm.model_name,
            'dataset': dataset.dataset_name,
            'samples': dataset.num_samples,
            f'{setting}': avg_score,
            'elapsed_sec': round(task_elapsed, 2),
            'gpu_mem_allocated_gib': round(gpu_mem_allocated, 2),
            'gpu_mem_reserved_gib': round(gpu_mem_reserved, 2),
            '_sample_details': sample_details,
            '_sample_scores': scores,
            '_math_verify_scores': math_verify_scores,
        }
        if dataset.dataset_name == 'math500' and math_verify_scores:
            stat[f'{setting}_math_verify'] = sum(math_verify_scores) / len(math_verify_scores)
        self.all_stats.append(stat)

        # Save per-task sample json
        if self.samples_dir and self.dist_config.master_process:
            self._write_samples_json(stat, output_path)

        if self.dist_config.is_distributed:
            dist.barrier()

    def _write_samples_json(self, stat, jsonl_path):
        """Write per-task sample details to samples/{task}.json."""
        ds_name = stat['dataset']
        task_name = ds_name.split('/')[-1] if '/' in ds_name else ds_name
        details = stat.get('_sample_details', [])
        scores = stat.get('_sample_scores', [])

        # Read predictions from jsonl
        jsonl_data = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf8') as f:
                for line in f:
                    try:
                        jsonl_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        samples = []
        for idx, detail in enumerate(details):
            sample = {
                'index': idx + 1,
                'score': scores[idx] if idx < len(scores) else None,
                'input_len': detail.get('input_len'),
                'raw_gen_len': detail.get('raw_gen_len'),
                'output_len': detail.get('output_len'),
                'kv_budget': detail.get('sparse_budget'),
                'total_kv_budget': detail.get('total_kv_budget'),
                'select_sets': detail.get('select_sets'),
            }
            if idx < len(jsonl_data):
                d = jsonl_data[idx]
                preds_list = d.get('prediction', d.get('response', []))
                gt_list = d.get('ground_truth', d.get('answer', []))
                sample['prediction'] = preds_list[0] if preds_list and isinstance(preds_list[0], str) else str(preds_list[0]) if preds_list else ''
                sample['ground_truth'] = gt_list
            samples.append(sample)

        out = {
            'task': ds_name,
            'num_samples': len(samples),
            'avg_score': sum(scores) / len(scores) if scores else 0.0,
            'samples': samples,
        }

        sample_path = os.path.join(self.samples_dir, f"{task_name}.json")
        with open(sample_path, 'w', encoding='utf8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    def summarize(self):
        # Exclude internal fields from DataFrame
        df_stats = [{k: v for k, v in s.items() if not k.startswith('_')} for s in self.all_stats]
        df = pd.DataFrame(df_stats)

        if self.dist_config.is_distributed:
            dist.barrier()
            output = [None for _ in range(self.dist_config.world_size)]
            dist.gather_object(df, output if self.dist_config.master_process else None, dst=0)
            dist.barrier()
            if self.dist_config.master_process:
                df = pd.concat(output)
                setting_columns = [col for col in df.columns if col not in ['model', 'dataset', 'samples',
                                   'elapsed_sec', 'gpu_mem_allocated_gib', 'gpu_mem_reserved_gib']]
                samples_sum = df.groupby(['model', 'dataset'])['samples'].sum()
                agg_dict = {col: 'mean' for col in setting_columns}
                weighted_means = df.groupby(['model', 'dataset']).apply(lambda x: pd.Series({
                    col: (x[col] * x['samples']).sum() / x['samples'].sum() for col in setting_columns
                }))
                df = weighted_means.join(samples_sum).reset_index()

        if self.dist_config.master_process:
            score_columns = [col for col in df.columns if col not in ['model', 'dataset', 'samples',
                             'elapsed_sec', 'gpu_mem_allocated_gib', 'gpu_mem_reserved_gib']]
            numeric_columns = df.select_dtypes(include='number')
            mean_values = numeric_columns.mean()
            mean_row = pd.DataFrame({col: [mean_values[col] if col in mean_values else 'mean'] for col in df.columns})
            df_with_mean = pd.concat([df, mean_row], ignore_index=True)
            print(df_with_mean.to_markdown(index=False))

            # Write summary.txt
            self._write_summary(df_with_mean)

            # Restore stderr
            if hasattr(self, '_tee_stderr'):
                sys.stderr = self._tee_stderr.stream
                self._tee_stderr.close()

    def _write_summary(self, results_df):
        """Write summary.txt in lm_eval style format."""
        if not self.run_dir:
            return

        total_elapsed = time.time() - self.run_start_time
        args = self.args

        # GPU info
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

        # Detect setting column
        setting_col = [c for c in results_df.columns if c not in ['model', 'dataset', 'samples', 'elapsed_sec', 'gpu_mem_allocated_gib', 'gpu_mem_reserved_gib']]
        setting_key = setting_col[0] if setting_col else 'score'

        lines = []
        sep = "=" * 70
        lines.append(sep)
        lines.append("ShadowKV Evaluation Log")
        lines.append(sep)
        lines.append("")
        lines.append(f"{'Timestamp:':<20s} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"{'Model:':<20s} {args.model_name if args else 'N/A'}")
        lines.append(f"{'Method:':<20s} {args.method if args else 'N/A'}")
        think_str = 'ON' if (args and args.enable_thinking) else 'OFF'
        lines.append(f"{'Thinking mode:':<20s} {think_str}")
        lines.append(f"{'Max context len:':<20s} {args.datalen if args else 'N/A'}")
        lines.append(f"{'Samples/dataset:':<20s} {args.num_samples if args else 'N/A'}")
        lines.append(f"{'Dtype:':<20s} bfloat16")
        lines.append("")

        # --- Hardware ---
        lines.append("--- Hardware ---")
        lines.append(f"{'GPU:':<20s} {gpu_name}")
        lines.append(f"{'Peak GPU memory:':<20s} {peak_gpu_mem:.2f} GB")
        lines.append(f"{'Total elapsed time:':<20s} {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        lines.append("")

        # --- KV Cache ---
        lines.append("--- KV Cache ---")
        lines.append(f"{'Sparse budget:':<20s} {args.sparse_budget if args else 'N/A'}")
        if args and args.sparse_budget_ratio:
            lines.append(f"{'Total KV ratio:':<20s} {args.sparse_budget_ratio}")
            # Compute effective sparse-only ratio from recorded details
            _pre_details = next((s.get('_sample_details', []) for s in self.all_stats if s.get('_sample_details')), [])
            if _pre_details:
                _pre_d = _pre_details[0]
                _cs = _pre_d.get('chunk_size', args.chunk_size if args else 8)
                _overhead = _pre_d.get('outlier_chunk', 0) * _cs + _pre_d.get('local_chunk', 0) * _cs
                _sparse_ratios = [d['sparse_budget'] / d['input_len'] for d in _pre_details if d.get('sparse_budget') and d.get('input_len', 0) > 0]
                if _sparse_ratios:
                    lines.append(f"{'Sparse-only ratio:':<20s} {sum(_sparse_ratios)/len(_sparse_ratios):.4f} (excl. outlier+local {_overhead} tokens)")
        # Get outlier/local info from recorded sample details
        _first_details = next((s.get('_sample_details', []) for s in self.all_stats if s.get('_sample_details')), [])
        _first_detail = _first_details[0] if _first_details else {}
        chunk_size = _first_detail.get('chunk_size', args.chunk_size if args else 8)
        outlier_chunk = _first_detail.get('outlier_chunk', 0)
        local_chunk = _first_detail.get('local_chunk', 0)
        outlier_tokens = outlier_chunk * chunk_size
        local_tokens = local_chunk * chunk_size
        lines.append(f"{'Outlier tokens:':<20s} {outlier_tokens} ({outlier_chunk} chunks x {chunk_size})")
        lines.append(f"{'Local tokens:':<20s} {local_tokens} ({local_chunk} chunks x {chunk_size})")
        lines.append(f"{'Rank:':<20s} {args.rank if args else 'N/A'}")
        lines.append(f"{'Chunk size:':<20s} {args.chunk_size if args else 'N/A'}")
        lines.append("")

        # --- Per-Dataset Timing ---
        lines.append("--- Per-Dataset Timing ---")
        for stat in self.all_stats:
            ds = stat['dataset']
            n = stat['samples']
            t = stat.get('elapsed_sec', 0)
            mem = stat.get('gpu_mem_allocated_gib', 0)
            lines.append(f"  {ds:<35s} {n} samples  {t:>8.1f}s  peak {mem:.2f}GB")
        lines.append("")

        # --- Results (lm_eval style table) ---
        lines.append("--- Results ---")
        lines.append("")

        # LongBench task groups and metric definitions
        LONGBENCH_GROUPS = {
            'Single-Document QA': {
                'tasks': ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh'],
                'metrics': {
                    'narrativeqa': 'qa_f1_score',
                    'qasper': 'qa_f1_score',
                    'multifieldqa_en': 'qa_f1_score',
                    'multifieldqa_zh': 'qa_f1_zh_score',
                },
            },
            'Multi-Document QA': {
                'tasks': ['hotpotqa', '2wikimqa', 'musique', 'dureader'],
                'metrics': {
                    'hotpotqa': 'qa_f1_score',
                    '2wikimqa': 'qa_f1_score',
                    'musique': 'qa_f1_score',
                    'dureader': 'rouge_zh_score',
                },
            },
            'Summarization': {
                'tasks': ['gov_report', 'qmsum', 'multi_news', 'vcsum'],
                'metrics': {
                    'gov_report': 'rouge_score',
                    'qmsum': 'rouge_score',
                    'multi_news': 'rouge_score',
                    'vcsum': 'rouge_zh_score',
                },
            },
            'Few-shot Learning': {
                'tasks': ['trec', 'triviaqa', 'samsum', 'lsht'],
                'metrics': {
                    'trec': 'classification_score',
                    'triviaqa': 'qa_f1_score',
                    'samsum': 'rouge_score',
                    'lsht': 'classification_score',
                },
            },
            'Synthetic Tasks': {
                'tasks': ['passage_count', 'passage_retrieval_en', 'passage_retrieval_zh'],
                'metrics': {
                    'passage_count': 'count_score',
                    'passage_retrieval_en': 'retrieval_score',
                    'passage_retrieval_zh': 'retrieval_zh_score',
                },
            },
            'Code Completion': {
                'tasks': ['lcc', 'repobench-p'],
                'metrics': {
                    'lcc': 'code_sim_score',
                    'repobench-p': 'code_sim_score',
                },
            },
        }

        def _get_metric_name(task_short, groups=LONGBENCH_GROUPS):
            for gname, ginfo in groups.items():
                if task_short in ginfo['metrics']:
                    return ginfo['metrics'][task_short]
            if task_short == 'math500':
                return 'exact_match'
            return 'score'

        def _compute_stderr(sample_scores):
            n = len(sample_scores)
            if n > 1:
                mean_s = sum(sample_scores) / n
                var = sum((x - mean_s) ** 2 for x in sample_scores) / (n - 1)
                return math.sqrt(var / n)
            return 0.0

        # Build lookup: task_short -> stat
        stat_by_task = {}
        for stat in self.all_stats:
            ds = stat['dataset']
            task_short = ds.split('/')[-1] if '/' in ds else ds
            stat_by_task[task_short] = stat

        header = f"|{'Tasks':^40s}|{'n-shot':>6s}|{'Metric':^20s}|   |{'Value':^7s}|   |{'Stderr':^7s}|"
        lines.append(header)
        lines.append(f"|{'-'*40}|{'-'*6}:|{'-'*20}|---|{'-'*7}:|---|{'-'*7}:|")

        group_summaries = []
        all_scores_flat = []

        for group_name, group_info in LONGBENCH_GROUPS.items():
            group_scores = []
            group_stderrs = []
            group_task_lines = []

            for task in group_info['tasks']:
                if task not in stat_by_task:
                    continue
                stat = stat_by_task[task]
                score = stat.get(setting_key, 0)
                metric_name = group_info['metrics'].get(task, 'score')
                sample_scores = stat.get('_sample_scores', [])
                stderr = _compute_stderr(sample_scores)

                all_scores_flat.extend(sample_scores)
                group_scores.append(score)
                group_stderrs.append(stderr)

                ds_label = f"longbench_{task}"
                group_task_lines.append(
                    f"| - {ds_label:<37s}|{0:>6d}|{metric_name:<20s}| ↑ |{score:.4f}| ± |{stderr:.4f}|"
                )

            if not group_scores:
                continue

            # Group average and pooled stderr
            group_avg = sum(group_scores) / len(group_scores)
            group_avg_stderr = sum(group_stderrs) / len(group_stderrs)
            group_summaries.append((group_name, group_avg, group_avg_stderr))

            # Group header line
            lines.append(f"|{'- ' + group_name:<40s}|{'':>6s}|{'score':<20s}| ↑ |{group_avg:.4f}| ± |{group_avg_stderr:.4f}|")
            for tl in group_task_lines:
                lines.append(tl)

        lines.append("")

        # Groups summary table
        lines.append(f"|{'Groups':^26s}|{'Metric':^8s}|   |{'Value':^7s}|   |{'Stderr':^7s}|")
        lines.append(f"|{'-'*26}|{'-'*8}|---|{'-'*7}:|---|{'-'*7}:|")
        for gname, gavg, gstderr in group_summaries:
            lines.append(f"|{'- ' + gname:<26s}|{'score':<8s}| ↑ |{gavg:.4f}| ± |{gstderr:.4f}|")
        lines.append("")

        # Average
        if all_scores_flat:
            avg_all = sum(all_scores_flat) / len(all_scores_flat)
        else:
            avg_all = 0.0
        lines.append(f"Average score (all datasets): {avg_all * 100:.1f}")
        lines.append("")

        # --- Inference Stats ---
        lines.append("--- Inference Stats ---")
        for stat in self.all_stats:
            ds = stat['dataset']
            details = stat.get('_sample_details', [])
            if not details:
                continue

            input_lens = [d.get('input_len', 0) for d in details]
            output_lens = [d.get('output_len', 0) for d in details]
            raw_gens = [d.get('raw_gen_len', 0) for d in details if d.get('raw_gen_len') is not None]
            kv_budgets = [d.get('sparse_budget', 0) for d in details if d.get('sparse_budget') is not None]
            total_kv_budgets = [d.get('total_kv_budget', 0) for d in details if d.get('total_kv_budget') is not None]
            sel_sets_list = [d.get('select_sets', 0) for d in details if d.get('select_sets') is not None]

            def _fmt_stat(vals):
                if not vals:
                    return "N/A"
                avg = sum(vals) / len(vals)
                return f"{int(avg)} [{min(vals)}-{max(vals)}]"

            def _fmt_ratio(kv_vals, in_vals):
                if not kv_vals or not in_vals:
                    return "N/A"
                ratios = [k / i for k, i in zip(kv_vals, in_vals) if i > 0]
                if not ratios:
                    return "N/A"
                return f"{sum(ratios)/len(ratios):.4f}"

            ds_short = ds.split('/')[-1] if '/' in ds else ds
            parts = [f"input_len={_fmt_stat(input_lens)}"]
            if raw_gens:
                parts.append(f"raw_gen={_fmt_stat(raw_gens)}")
            parts.append(f"output_len={_fmt_stat(output_lens)}")
            if kv_budgets:
                parts.append(f"selected_kv={_fmt_stat(kv_budgets)}")
                parts.append(f"ratio={_fmt_ratio(kv_budgets, input_lens)}")
            if total_kv_budgets:
                parts.append(f"total_kv={_fmt_stat(total_kv_budgets)}")
                parts.append(f"total_ratio={_fmt_ratio(total_kv_budgets, input_lens)}")
            if sel_sets_list:
                parts.append(f"select_sets={_fmt_stat(sel_sets_list)}")

            lines.append(f"  {ds_short + ':':<15s} {'  '.join(parts)}")

        lines.append("")
        lines.append(sep)

        summary_path = os.path.join(self.run_dir, "summary.txt")
        with open(summary_path, 'w', encoding='utf8') as f:
            f.write("\n".join(lines))

        print(colored(f"\n[Log] Saved to {self.run_dir}/", 'green'))
