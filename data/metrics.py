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

import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str

def string_match_part(preds, refs):
    preds = postprocess_pred(preds)
    if isinstance(refs, str):
        refs = [refs]
    score_ref_in_pred = max([1.0 if r.lower() in preds.lower() else 0.0 for r in refs])
    score_pred_in_ref = max([1.0 if preds.lower() in r.lower() else 0.0 for r in refs])
    score = max(score_ref_in_pred, score_pred_in_ref)
    return round(score, 2)

def multi_number(prediction: str, ground_truth: list) -> float:
    assert type(prediction) == str, f"Prediction is not a string, but {prediction}, type: {type(prediction)}"
    assert type(ground_truth) == list, f"Ground truth is not a list, but {ground_truth}, type: {type(ground_truth)}"
    prediction = normalize_answer(prediction)
    prediction_list = re.findall(r'\d+', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    hit_rate = len(hits) / len(ground_truth)
    
    return hit_rate

def multi_words(prediction: str, ground_truth: list) -> float:
    prediction = prediction.lower()
    ground_truth = [gt.lower() for gt in ground_truth]
    prediction_list = re.findall(r'\b\w+\b', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    hit_rate = len(hits) / len(ground_truth)
    
    return hit_rate

def needle_score(prediction, ground_truth):
    assert type(prediction) == str, f"Prediction is not a string, but {prediction}, type: {type(prediction)}"
    assert type(ground_truth) == str, f"Ground truth is not a string, but {ground_truth}, type: {type(ground_truth)}"
    prediction = normalize_answer(postprocess_pred(prediction))
    ground_truth = normalize_answer(ground_truth)
    min_length = min(len(prediction), len(ground_truth))
    min_length = len(ground_truth)
    score =  float((prediction[:min_length] == ground_truth[:min_length]))
    pred_list = prediction.split()
    score = max(float(ground_truth in pred_list), score)
    return score


# ============================================================
# LongBench metrics
# ============================================================

def _f1_score(prediction, ground_truth):
    """Token-level F1 score."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth):
    """QA F1 score used by LongBench for QA tasks."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return _f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction, ground_truth):
    """ROUGE-L F1 score for summarization tasks."""
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("Please install rouge: pip install rouge")
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth):
    """ROUGE-L score for Chinese summarization tasks."""
    try:
        import jieba
    except ImportError:
        raise ImportError("Please install jieba: pip install jieba")
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, all_classes=None):
    """Classification score for few-shot classification tasks (trec, lsht)."""
    if all_classes is None:
        return float(ground_truth.lower() in prediction.lower())
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def count_score(prediction, ground_truth):
    """Score for passage_count task."""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_score(prediction, ground_truth):
    """Score for passage_retrieval_en task."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction, ground_truth):
    """Score for passage_retrieval_zh task."""
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def code_sim_score(prediction, ground_truth):
    """Code similarity score for code completion tasks."""
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        raise ImportError("Please install fuzzywuzzy: pip install fuzzywuzzy python-Levenshtein")
    all_lines = prediction.lstrip("\n").split("\n")
    prediction_line = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction_line = line
            break
    return fuzz.ratio(prediction_line, ground_truth) / 100


def longbench_metric(prediction, ground_truth, task_name, all_classes=None):
    """Unified LongBench metric dispatcher.

    Args:
        prediction: model prediction string
        ground_truth: list of reference answers
        task_name: LongBench subtask name
        all_classes: class list for classification tasks (trec, lsht)

    Returns:
        float: score
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    prediction = postprocess_pred(prediction).strip()

    # Select metric function based on task
    if task_name in ['hotpotqa', '2wikimqa', 'musique', 'narrativeqa', 'qasper',
                     'multifieldqa_en', 'triviaqa']:
        return max(qa_f1_score(prediction, gt) for gt in ground_truth)
    elif task_name in ['multifieldqa_zh', 'dureader']:
        return max(rouge_zh_score(prediction, gt) for gt in ground_truth)
    elif task_name in ['gov_report', 'qmsum', 'multi_news', 'samsum']:
        return max(rouge_score(prediction, gt) for gt in ground_truth)
    elif task_name == 'vcsum':
        return max(rouge_zh_score(prediction, gt) for gt in ground_truth)
    elif task_name in ['trec', 'lsht']:
        return max(classification_score(prediction, gt, all_classes=all_classes) for gt in ground_truth)
    elif task_name == 'passage_count':
        return max(count_score(prediction, gt) for gt in ground_truth)
    elif task_name == 'passage_retrieval_en':
        return max(retrieval_score(prediction, gt) for gt in ground_truth)
    elif task_name == 'passage_retrieval_zh':
        return max(retrieval_zh_score(prediction, gt) for gt in ground_truth)
    elif task_name in ['lcc', 'repobench-p']:
        return max(code_sim_score(prediction, gt) for gt in ground_truth)
    else:
        # Fallback to QA F1
        return max(qa_f1_score(prediction, gt) for gt in ground_truth)


# ============================================================
# MATH500 metrics
# ============================================================

def extract_boxed_answer(text):
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    # Find all \boxed{...} patterns
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None

    # Find matching closing brace
    depth = 0
    start = idx + len('\\boxed{')
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def normalize_math_answer(answer):
    """Normalize a math answer string for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    # Remove \text{}, \mathrm{}, etc.
    answer = re.sub(r'\\(text|mathrm|textbf)\{([^}]*)\}', r'\2', answer)
    # Remove \left, \right
    answer = answer.replace('\\left', '').replace('\\right', '')
    # Remove display style
    answer = answer.replace('\\displaystyle', '')
    # Remove spaces
    answer = answer.replace(' ', '')
    # Remove trailing period
    answer = answer.rstrip('.')
    return answer


def math_exact_match(prediction, ground_truth):
    """Exact match metric for MATH500.

    Extracts \\boxed{} answer from prediction, compares with ground truth.
    Falls back to checking if the answer appears at the end of the prediction.
    """
    prediction = postprocess_pred(prediction)

    # If thinking tokens present, only use content after </think>
    if '</think>' in prediction:
        prediction = prediction.split('</think>')[-1].strip()

    # Try to extract boxed answer
    pred_answer = extract_boxed_answer(prediction)

    if pred_answer is None:
        # Fallback: try to find the answer in the last line
        lines = prediction.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        # Check if ground truth appears in last line
        norm_last = normalize_math_answer(last_line)
        norm_gt = normalize_math_answer(ground_truth)
        if norm_gt and norm_gt in norm_last:
            return 1.0
        return 0.0

    norm_pred = normalize_math_answer(pred_answer)
    norm_gt = normalize_math_answer(ground_truth)

    if norm_pred == norm_gt:
        return 1.0

    # Try sympy equivalence check
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
        pred_expr = parse_latex(pred_answer)
        gt_expr = parse_latex(ground_truth)
        if simplify(pred_expr - gt_expr) == 0:
            return 1.0
    except Exception:
        pass

    return 0.0


# ============================================================
# InfiniteBench metrics
# ============================================================

def infinitebench_metric(prediction, ground_truth, task_name):
    """Unified InfiniteBench metric dispatcher."""
    prediction = postprocess_pred(prediction).strip()

    if task_name in ('passkey', 'number_string', 'kv_retrieval'):
        # Exact substring match
        answer = str(ground_truth).strip()
        return 1.0 if answer in prediction else 0.0

    elif task_name == 'longbook_qa_eng':
        # F1 score
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        best = 0.0
        for gt in gts:
            pred_tokens = normalize_answer(prediction).split()
            gt_tokens = normalize_answer(gt).split()
            if pred_tokens and gt_tokens:
                best = max(best, _f1_score(pred_tokens, gt_tokens))
        return best

    elif task_name == 'longbook_qa_chn':
        # Chinese F1 score with jieba
        try:
            import jieba
        except ImportError:
            return 0.0
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punc = set(string.punctuation + cn_punc)
        def _norm_zh(s):
            return "".join(ch for ch in s.lower() if ch not in all_punc).replace(" ", "")

        best = 0.0
        for gt in gts:
            pred_tokens = [_norm_zh(t) for t in jieba.cut(prediction, cut_all=False)]
            gt_tokens = [_norm_zh(t) for t in jieba.cut(gt, cut_all=False)]
            pred_tokens = [t for t in pred_tokens if t]
            gt_tokens = [t for t in gt_tokens if t]
            if pred_tokens and gt_tokens:
                best = max(best, _f1_score(pred_tokens, gt_tokens))
        return best

    elif task_name in ('longbook_choice_eng', 'code_debug'):
        # Multiple choice: check A/B/C/D
        answer = ground_truth
        if isinstance(answer, list):
            correct_letter = answer[1] if len(answer) > 1 else answer[0]
            correct_text = answer[0]
        else:
            correct_letter = answer
            correct_text = answer
        pred_upper = prediction.upper().strip()
        if correct_letter.upper() in pred_upper[:5]:
            return 1.0
        if correct_text.lower() in prediction.lower():
            return 1.0
        return 0.0

    elif task_name == 'longbook_sum_eng':
        # ROUGE-L
        if not prediction or not str(ground_truth):
            return 0.0
        return rouge_score(prediction, str(ground_truth))

    elif task_name == 'longdialogue_qa_eng':
        # Character name match
        answer = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        return 1.0 if answer.lower() in prediction.lower() else 0.0

    elif task_name == 'math_find':
        # First integer match
        answer = str(ground_truth).strip()
        pred_nums = re.split(r"[^0-9]", prediction)
        for item in pred_nums:
            if item:
                return 1.0 if item == answer else 0.0
        return 0.0

    else:
        # Fallback: substring match
        return 1.0 if str(ground_truth).lower() in prediction.lower() else 0.0


def math_verify_score(prediction, ground_truth):
    """math_verify metric for MATH500.

    Uses the math_verify library for robust mathematical equivalence checking.
    Handles equivalent expressions like 1/2 vs 0.5, different LaTeX forms, etc.
    """
    from math_verify import parse, verify

    prediction = postprocess_pred(prediction)

    # If thinking tokens present, only use content after </think>
    if '</think>' in prediction:
        prediction = prediction.split('</think>')[-1].strip()

    try:
        gold = parse(f"$\\boxed{{{ground_truth}}}$")
    except Exception:
        return 0.0

    # Try to extract \boxed{} from prediction
    pred_answer = extract_boxed_answer(prediction)
    if pred_answer is not None:
        try:
            pred = parse(f"${pred_answer}$")
            if verify(gold, pred):
                return 1.0
        except Exception:
            pass

    # Fallback: try parsing the entire prediction
    try:
        pred = parse(f"${prediction}$")
        if verify(gold, pred):
            return 1.0
    except Exception:
        pass

    return 0.0
