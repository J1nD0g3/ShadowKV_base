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

from datasets import load_dataset
from termcolor import colored
import random
import numpy as np

# RULER
from .metrics import needle_score, string_match_part, multi_number, multi_words
from .metrics import longbench_metric, math_exact_match

# NIAH
from data.utils import generate_random_number, read_context_files, create_contexts, NIAH_TEMPLATE, RANDOM_NEEDLE_CITIES

METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
}

GEN_LEN = {
    'niah': 64,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32,
}

DATADIR = {
    'ruler': 'data/ruler/data',
    'niah': 'data/niah/data',
}

# LongBench prompt templates per subtask
LONGBENCH_PROMPTS = {
    'narrativeqa': "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    'qasper': "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    'multifieldqa_en': "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    'multifieldqa_zh': "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n答案：",
    'hotpotqa': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    '2wikimqa': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    'musique': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    'dureader': "请根据以下给定文章回答问题，只需给出答案，不要输出其他任何字词。\n\n{context}\n\n问题：{input}\n答案：",
    'gov_report': "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    'qmsum': "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nQuery: {input}\n\nAnswer:",
    'multi_news': "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    'vcsum': "下面有一段会议记录，请用中文总结会议的内容。\n\n会议记录：\n{context}\n\n总结：",
    'trec': "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    'triviaqa': "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    'samsum': "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    'lsht': "请根据所给的几个例子判断下面文章的类别。\n\n{context}\n{input}",
    'passage_count': "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is:",
    'passage_retrieval_en': "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is:",
    'passage_retrieval_zh': '以下是若干段落和一个摘要，请确定摘要来自哪个段落。\n\n{context}\n\n以下是摘要。\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\u201c段落1\u201d、\u201c段落2\u201d等。\n\n答案是：',
    'lcc': "Please complete the code given below. \n{context}Next line of code:",
    'repobench-p': "Please complete the code given below. \n{context}Next line of code:",
}

LONGBENCH_GEN_LEN = {
    'narrativeqa': 128, 'qasper': 128,
    'multifieldqa_en': 64, 'multifieldqa_zh': 64,
    'hotpotqa': 32, '2wikimqa': 32, 'musique': 32, 'dureader': 128,
    'gov_report': 512, 'qmsum': 512, 'multi_news': 512, 'vcsum': 512,
    'trec': 64, 'triviaqa': 32, 'samsum': 128, 'lsht': 64,
    'passage_count': 32, 'passage_retrieval_en': 32, 'passage_retrieval_zh': 32,
    'lcc': 64, 'repobench-p': 64,
}

# MATH500 few-shot prompt (4-shot, same as minerva_math)
MATH500_FEWSHOT_PROMPT = """Problem:
Find the domain of the expression  $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge 2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.

Problem:
If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$

Solution:
We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}.$

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he digit uses two 15-pound weights instead, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Setting $480=30n$, we find $n=\\boxed{{16}}$.

Problem:
If the system of equations

\\begin{{align*}}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain

$$6y-9x=-\\frac{{3}}{{2}}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$$

Problem:
{problem}

Solution:
"""

class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1, enable_thinking=False, disable_chat_template=False):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.disable_chat_template = disable_chat_template
        self.world_size = world_size
        self.is_sharded = False
        self.all_classes = []  # For LongBench classification tasks
        self.enable_thinking = enable_thinking

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()

        self.num_samples = len(self.tokenized_prompts)
        self.gen_len = self.get_gen_len()
        self.metric = self.get_metric()

    def _apply_chat_template(self, prompt_text):
        """Apply chat template to prompt text and return tokenized input_ids.

        For Qwen3: always use apply_chat_template with enable_thinking=True
        to get a clean prompt (no injected <think></think> tokens).
        The generate() method handles thinking mode separately.
        For other models: use prompt_template.py Templates.
        """
        if self.disable_chat_template:
            return self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
        model_name = self.tokenizer.name_or_path.lower()
        if 'qwen3' in model_name:
            messages = [{"role": "user", "content": prompt_text}]
            # enable_thinking=True: clean prompt, model generates <think>...</think> then answer
            # enable_thinking=False: injects <think>\n\n</think>\n\n to skip thinking
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            return self.tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)
        else:
            from models.prompt_template import Templates
            if 'qwen' in model_name:
                template_key = 'qwen'
            elif 'llama-3' in model_name or 'llama3' in model_name:
                template_key = 'llama-3'
            elif 'yi' in model_name:
                template_key = 'yi'
            elif 'glm' in model_name:
                template_key = 'glm'
            elif 'lwm' in model_name:
                template_key = 'lwm'
            elif 'phi' in model_name:
                template_key = 'phi'
            else:
                template_key = 'base'
            chat_text = Templates[template_key].format(ctx=prompt_text)
            return self.tokenizer.encode(chat_text, return_tensors="pt", add_special_tokens=False)

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            if self.all_classes:
                self.all_classes = self.all_classes[start:end]
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        # Base gen_len per dataset (without thinking budget).
        # When enable_thinking, evaluator dynamically caps gen_len to
        # max_length - input_len, giving the model remaining capacity.
        if self.dataset_name.startswith('longbench/'):
            task = self.dataset_name.split('/')[-1]
            base = LONGBENCH_GEN_LEN.get(task, 64)
        elif self.dataset_name == 'math500':
            base = 4096
        elif 'niah' == self.dataset_name:
            base = 10
        elif 'niah' in self.dataset_name:
            base = 128
        elif 'vt' in self.dataset_name:
            base = 30
        elif 'cwe' in self.dataset_name:
            base = 120
        elif 'fwe' in self.dataset_name:
            base = 50
        elif 'qa' in self.dataset_name:
            base = 32
        else:
            raise Exception("Gen len not found")

        if self.enable_thinking:
            # Use max_length as upper bound; evaluator will clamp per sample
            return self.datalen  # large ceiling, evaluator handles actual cap
        return base

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    def get_metric(self):
        if self.dataset_name.startswith('longbench/'):
            task = self.dataset_name.split('/')[-1]
            # Return a lambda that captures the task name
            def _lb_metric(pred, gt, all_classes=None):
                return longbench_metric(pred, gt, task, all_classes=all_classes)
            return _lb_metric
        elif self.dataset_name == 'math500':
            return math_exact_match
        elif 'multiquery' in self.dataset_name or 'multivalue' in self.dataset_name:
            return METRICS_FN['multi']
        elif 'niah' in self.dataset_name:
            return METRICS_FN['niah']
        elif 'vt' in self.dataset_name:
            return METRICS_FN['vt']
        elif 'cwe' in self.dataset_name:
            return METRICS_FN['cwe']
        elif 'fwe' in self.dataset_name:
            return METRICS_FN['fwe']
        elif 'qa' in self.dataset_name:
            return METRICS_FN['qa']
        else:
            raise Exception("Metric not found")

    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]
            assert self.datalen in [8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024], "Only support datalen of 16k, 32k, 64k, 128k"

            if 'llama-3' in self.tokenizer.name_or_path.lower():
                model_dir = 'llama-3'
            elif 'yi' in self.tokenizer.name_or_path.lower():
                model_dir = 'yi'
            elif 'lwm' in self.tokenizer.name_or_path.lower():
                model_dir = 'lwm'
            elif 'glm' in self.tokenizer.name_or_path.lower():
                model_dir = 'glm'
            elif 'qwen3' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen3'
            elif 'qwen' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen'
            elif 'phi' in self.tokenizer.name_or_path.lower():
                model_dir = 'phi'
            else:
                raise Exception("Model not found", self.tokenizer.name_or_path)

            dataset = load_dataset("json", data_files=f'{DATADIR["ruler"]}/{model_dir}/{self.datalen}/{task}/validation.jsonl', split='train')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []

            for i in range(self.num_samples):
                input_text = dataset[i]['input']
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                tokenized_prompts.append(input_ids)
                gt.append(dataset[i]['outputs'])

            return tokenized_prompts, gt

        elif self.dataset_name == 'niah':
            print(colored(f"[Warning] NIAH dataset cannot set # samples, it is up to world_size, which is set to {self.world_size}", 'red'))
            
            haystack_file = f'{DATADIR["niah"]}/pg19_mini.jsonl'
            context_lengths_min = 16*1024
            context_lengths_max = self.datalen
            n_context_length_intervals = 15
            n_document_depth_intervals = 10  # position of the needle in the haystack
            n_rounds = 1 # max(1, 4 // self.world_size) # 8 rounds in total assume we have 8xGPUs
            needle = "\nThe special magic {city} number is: {rnd_number}\n"
            retrieval_question="What is the special magic {} number?"
            rnd_number_digits = 7

            context_lengths = np.round(
                np.linspace(
                    context_lengths_min,
                    context_lengths_max,
                    num=n_context_length_intervals,
                    endpoint=True,
                )
            ).astype(int)

            document_depth_percents = np.round( # we use linear scale here
                np.linspace(
                    0,
                    100,
                    num=n_document_depth_intervals,
                    endpoint=True,
                )
            ).astype(int)

            self.is_sharded = True # we shard the data during init dataset
            
            full_contexts = read_context_files(n=n_rounds, context_lengths=context_lengths, haystack_file=haystack_file, tokenizer=self.tokenizer)
            full_tokens = [
                self.tokenizer.encode(full_context, add_special_tokens=False) for full_context in full_contexts
            ]

            tokenized_prompts = []
            gt = []
            ctx_len = []
            depth_pct = []

            for context_length in context_lengths:
                trim_contexts = [
                    self.tokenizer.decode(full_token[:context_length], skip_special_tokens=True)
                    for full_token in full_tokens
                ]
                contexts = []
                for depth_percent in document_depth_percents:
                    for i in range(n_rounds):
                        random_city = random.choice(RANDOM_NEEDLE_CITIES)
                        insert_needle = True
                        needle_rnd_number = str(generate_random_number(rnd_number_digits))
                        context = create_contexts(
                            needle_rnd_number=needle_rnd_number,
                            insert_needle=insert_needle,
                            random_city=random_city,
                            trim_context=trim_contexts[i],
                            context_length=context_length,
                            depth_percent=depth_percent,
                            needle=needle,
                            retrieval_question=retrieval_question,
                            tokenizer=self.tokenizer,
                            final_context_length_buffer=32,
                        )
                        contexts.append(context)

                for context in contexts:
                    prompt = NIAH_TEMPLATE.format(
                        context=context["context"], question=context["question"]
                    )
                    input_tensor = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
                    tokenized_prompts.append(input_tensor.input_ids)
                    gt.append(context["needle_rnd_number"])
                    ctx_len.append(context["context_length"])
                    depth_pct.append(context["depth_percent"])
            
            return tokenized_prompts, gt, ctx_len, depth_pct

        elif self.dataset_name.startswith('longbench/'):  # longbench/hotpotqa, longbench/narrativeqa, etc.
            task = self.dataset_name.split('/')[-1]
            assert task in LONGBENCH_PROMPTS, f"Unknown LongBench task: {task}. Available: {list(LONGBENCH_PROMPTS.keys())}"

            dataset = load_dataset("THUDM/LongBench", task, split='test')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)

            tokenized_prompts = []
            gt = []
            self.all_classes = []

            for i in range(self.num_samples):
                sample = dataset[i]
                prompt_template = LONGBENCH_PROMPTS[task]
                context = sample['context']
                question = sample['input']

                # Build full prompt and check length; if too long, truncate context
                prompt_text = prompt_template.format(context=context, input=question)
                input_ids = self._apply_chat_template(prompt_text)
                if input_ids.size(1) > self.datalen:
                    # Measure overhead (template + question tokens) by building prompt without context
                    empty_prompt = prompt_template.format(context='', input=question)
                    overhead_ids = self._apply_chat_template(empty_prompt)
                    max_ctx_tokens = self.datalen - overhead_ids.size(1)
                    # Tokenize context separately and truncate
                    ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
                    truncated_ctx = self.tokenizer.decode(ctx_ids[:max_ctx_tokens], skip_special_tokens=False)
                    prompt_text = prompt_template.format(context=truncated_ctx, input=question)
                    input_ids = self._apply_chat_template(prompt_text)
                tokenized_prompts.append(input_ids)
                gt.append(sample['answers'])
                self.all_classes.append(sample.get('all_classes', None))

            return tokenized_prompts, gt

        elif self.dataset_name == 'math500':
            dataset = load_dataset("HuggingFaceH4/MATH-500", split='test')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)

            tokenized_prompts = []
            gt = []

            for i in range(self.num_samples):
                sample = dataset[i]
                prompt_text = MATH500_FEWSHOT_PROMPT.format(problem=sample['problem'])
                input_ids = self._apply_chat_template(prompt_text)
                tokenized_prompts.append(input_ids)
                gt.append(sample['answer'])

            return tokenized_prompts, gt

        else:
            raise ValueError(f"Dataset {self.dataset_name} not found, please choose from: ruler/*, niah, longbench/*, math500")