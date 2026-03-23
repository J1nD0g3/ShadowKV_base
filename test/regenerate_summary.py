#!/usr/bin/env python3
"""Regenerate summary.txt from existing sample JSON files using the updated format."""
import json
import math
import os
import sys

def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not run_dir or not os.path.isdir(run_dir):
        print(f"Usage: python {sys.argv[0]} <run_dir>")
        sys.exit(1)

    samples_dir = os.path.join(run_dir, 'samples')
    old_summary = os.path.join(run_dir, 'summary.txt')

    # Parse header info from old summary
    header_lines = []
    timing_lines = []
    inference_lines = []
    section = 'header'
    if os.path.exists(old_summary):
        with open(old_summary, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if line == '--- Per-Dataset Timing ---':
                    section = 'timing'
                    timing_lines.append(line)
                    continue
                elif line == '--- Results ---':
                    section = 'results'
                    continue
                elif line == '--- Inference Stats ---':
                    section = 'inference'
                    inference_lines.append(line)
                    continue

                if section == 'header':
                    header_lines.append(line)
                elif section == 'timing':
                    timing_lines.append(line)
                elif section == 'inference':
                    inference_lines.append(line)

    # LongBench groups
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

    # Load sample data
    task_data = {}
    for fname in sorted(os.listdir(samples_dir)):
        if not fname.endswith('.json'):
            continue
        task_short = fname.replace('.json', '')
        with open(os.path.join(samples_dir, fname), 'r') as f:
            data = json.load(f)
        scores = [s['score'] for s in data['samples'] if s.get('score') is not None]
        task_data[task_short] = {
            'avg_score': data.get('avg_score', sum(scores) / len(scores) if scores else 0),
            'scores': scores,
            'samples': data['samples'],
        }

    def _compute_stderr(scores):
        n = len(scores)
        if n > 1:
            mean_s = sum(scores) / n
            var = sum((x - mean_s) ** 2 for x in scores) / (n - 1)
            return math.sqrt(var / n)
        return 0.0

    # Build results table
    results_lines = []
    results_lines.append("--- Results ---")
    results_lines.append("")
    header = f"|{'Tasks':^40s}|{'n-shot':>6s}|{'Metric':^20s}|   |{'Value':^7s}|   |{'Stderr':^7s}|"
    results_lines.append(header)
    results_lines.append(f"|{'-'*40}|{'-'*6}:|{'-'*20}|---|{'-'*7}:|---|{'-'*7}:|")

    group_summaries = []
    all_scores_flat = []

    for group_name, group_info in LONGBENCH_GROUPS.items():
        group_scores = []
        group_stderrs = []
        group_task_lines = []

        for task in group_info['tasks']:
            if task not in task_data:
                continue
            td = task_data[task]
            score = td['avg_score']
            metric_name = group_info['metrics'].get(task, 'score')
            stderr = _compute_stderr(td['scores'])

            all_scores_flat.extend(td['scores'])
            group_scores.append(score)
            group_stderrs.append(stderr)

            ds_label = f"longbench_{task}"
            group_task_lines.append(
                f"| - {ds_label:<37s}|{0:>6d}|{metric_name:<20s}| ↑ |{score:.4f}| ± |{stderr:.4f}|"
            )

        if not group_scores:
            continue

        group_avg = sum(group_scores) / len(group_scores)
        group_avg_stderr = sum(group_stderrs) / len(group_stderrs)
        group_summaries.append((group_name, group_avg, group_avg_stderr))

        results_lines.append(f"|{'- ' + group_name:<40s}|{'':>6s}|{'score':<20s}| ↑ |{group_avg:.4f}| ± |{group_avg_stderr:.4f}|")
        for tl in group_task_lines:
            results_lines.append(tl)

    results_lines.append("")

    # Groups summary table
    results_lines.append(f"|{'Groups':^26s}|{'Metric':^8s}|   |{'Value':^7s}|   |{'Stderr':^7s}|")
    results_lines.append(f"|{'-'*26}|{'-'*8}|---|{'-'*7}:|---|{'-'*7}:|")
    for gname, gavg, gstderr in group_summaries:
        results_lines.append(f"|{'- ' + gname:<26s}|{'score':<8s}| ↑ |{gavg:.4f}| ± |{gstderr:.4f}|")
    results_lines.append("")

    if all_scores_flat:
        avg_all = sum(all_scores_flat) / len(all_scores_flat)
    else:
        avg_all = 0.0
    results_lines.append(f"Average score (all datasets): {avg_all * 100:.1f}")

    # Reconstruct summary
    output_lines = []
    output_lines.extend(header_lines)
    if timing_lines:
        output_lines.extend(timing_lines)
    output_lines.append("")
    output_lines.extend(results_lines)
    output_lines.append("")
    if inference_lines:
        output_lines.extend(inference_lines)
    output_lines.append("")
    output_lines.append("=" * 70)

    with open(old_summary, 'w', encoding='utf8') as f:
        f.write('\n'.join(output_lines))

    print(f"Regenerated {old_summary}")

if __name__ == '__main__':
    main()
