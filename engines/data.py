# -*- coding: utf-8 -*-
# @Time : 2023/7/10 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
from transformers import AutoTokenizer, LlamaTokenizer, BloomTokenizerFast
from transformers import DataCollatorWithPadding
from engines.utils.prompt_template import Template
from datasets import load_dataset
from itertools import chain
from glob import glob
import os


class DataManager:
    def __init__(self, config, logger):
        self.logger = logger
        self.mode = config.mode
        self.data_args = config.data_args
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.prompt_template = Template(self.data_args.prompt_template)
        logger.info(f'Load tokenizer from {self.model_args.model_path}')
        self.tokenizer = self.load_tokenizer(self.model_args.model_path)
        self.logger.info(f'Tokenizer:\n{self.tokenizer}')
        if self.data_args.ignore_pad_token_for_loss:
            self.label_pad_token_id = -100
        else:
            self.label_pad_token_id = self.tokenizer.pad_token_id
        self.use_firefly_loss = self.training_args.use_firefly_loss

    def load_tokenizer(self, model_path):
        if self.model_args.model_type in ['chatglm', 'baichuan', 'internlm', 'aquila', 'moss', 'xverse', 'mistral', 'yi']:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        elif self.model_args.model_type == 'qwen':
            # https://github.com/QwenLM/Qwen/issues/24
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, allowed_special='all')
        elif self.model_args.model_type == 'falcon':
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif self.model_args.model_type == 'rwkv':
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif self.model_args.model_type == 'bloom':
            tokenizer = BloomTokenizerFast.from_pretrained(model_path)
        elif self.model_args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=self.model_args.use_fast_tokenizer,
                                                       padding_side=self.model_args.padding_side)
        else:
            raise

        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = '<|endoftext|>'
            self.logger.info('Add eos token: {}'.format(tokenizer.eos_token))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info('Add pad token: {}'.format(tokenizer.pad_token))
        tokenizer.add_special_tokens(dict(additional_special_tokens=self.prompt_template.stop_words),
                                     replace_additional_special_tokens=False)
        return tokenizer

    def load_datasets_from_files(self, test=False):
        data_files = {}
        kwargs = {}
        if not test:
            if self.data_args.train_file_dir is not None and os.path.exists(self.data_args.train_file_dir):
                train_data_files = glob(f'{self.data_args.train_file_dir}/**/*.txt', recursive=True) + glob(
                    f'{self.data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                    f'{self.data_args.train_file_dir}/**/*.jsonl', recursive=True)
                self.logger.info(f"train files: {', '.join(train_data_files)}")
                data_files['train'] = train_data_files
            if self.training_args.do_eval and self.data_args.validation_file_dir is not None \
                    and os.path.exists(self.data_args.validation_file_dir):
                eval_data_files = glob(f'{self.data_args.validation_file_dir}/**/*.txt', recursive=True) + glob(
                    f'{self.data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                    f'{self.data_args.validation_file_dir}/**/*.jsonl', recursive=True)
                self.logger.info(f"eval files: {', '.join(eval_data_files)}")
                data_files['validation'] = eval_data_files
            extension = 'text' if data_files['train'][0].endswith('txt') else 'json'
            if extension == 'text':
                kwargs['keep_linebreaks'] = True
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                **kwargs
            )
            if self.training_args.do_eval and 'validation' not in raw_datasets.keys() \
                    and self.data_args.dev_ratio > 0.0:
                raw_datasets['validation'] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f'train[:{self.data_args.dev_ratio}%]',
                    cache_dir=self.model_args.cache_dir,
                    **kwargs
                )
                raw_datasets['train'] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f'train[{self.data_args.dev_ratio}%:]',
                    cache_dir=self.model_args.cache_dir,
                    **kwargs
                )
        else:
            if self.data_args.test_file is not None and os.path.exists(self.data_args.test_file):
                test_data_files = glob(
                    f'{self.data_args.test_file}/**/*.json', recursive=True) + glob(
                    f'{self.data_args.test_file}/**/*.jsonl', recursive=True)
                self.logger.info(f"test files: {', '.join(test_data_files)}")
                data_files['test'] = test_data_files
            raw_datasets = load_dataset(
                'json',
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
            )
        self.logger.info(f'Raw datasets: {raw_datasets}')
        return raw_datasets

    def format_example(self, examples, join_history=True):
        for i in range(len(examples['instruction'])):
            if examples['instruction'][i] and examples['output'][i]:
                query, answer = examples['instruction'][i], examples['output'][i]
                query = query + examples['input'][i] if examples['input'][i] else query
                if 'history' in examples and (history := examples['history'][i]) is not None:
                    prompt = self.prompt_template.get_prompt(query, history, join_history)
                else:
                    prompt = self.prompt_template.get_prompt(query, [], join_history)
                yield prompt, answer

    def transfer_front_tail_to_label_pad_token_id(self, label):
        start_pointer = 0
        end_pointer = len(label) - 1
        while label[start_pointer] != self.label_pad_token_id:
            label[start_pointer] = self.label_pad_token_id
            start_pointer += 1
        while label[end_pointer] != self.label_pad_token_id:
            label[end_pointer] = self.label_pad_token_id
            end_pointer -= 1
        return label

    def preprocess_pretrain_dataset(self, examples):
        # refer from https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/training/run_clm_pt_with_peft.py#L491
        tokenized_examples = self.tokenizer(examples['text'])
        block_size = self.data_args.max_input_token
        if block_size > self.tokenizer.model_max_length:
            self.logger.warning(
                f'The block_size passed ({block_size}) is larger than the maximum length for the model'
                f'({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}.'
            )
        block_size = min(block_size, self.tokenizer.model_max_length)

        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(tokenized_examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def preprocess_train_supervised_fine_tuning_dataset(self, examples):
        # ChatGLM1: https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py#L323
        # ChatGLM2: https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenization_chatglm.py#L171
        # Baichuan: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/tokenization_baichuan.py#L152
        # internlm: https://huggingface.co/internlm/internlm-chat-7b/blob/main/tokenization_internlm.py#L179
        # moss: https://huggingface.co/fnlp/moss-moon-003-sft/blob/main/tokenization_moss.py#L226
        # Llama: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L296
        inputs_list = []
        attention_mask_list = []
        labels_list = []
        if self.training_args.use_firefly_loss:
            for prompt, answer in self.format_example(examples, False):
                source_ids = []
                labels = []
                for i, sentence in enumerate(prompt):
                    if i % 2 == 0:
                        sentence_ids = self.tokenizer.encode(text=sentence, add_special_tokens=False)
                        source_ids.extend(sentence_ids)
                        labels.extend([self.label_pad_token_id] * (len(sentence_ids)))
                    else:
                        sentence_ids = self.tokenizer.encode(text=sentence, add_special_tokens=False)
                        sentence_ids = sentence_ids + [self.tokenizer.eos_token_id]
                        source_ids.extend(sentence_ids)
                        labels.extend(sentence_ids)
                target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
                if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
                    labels = self.tokenizer.build_inputs_with_special_tokens(labels)
                    context_length = len(labels)
                    labels = self.transfer_front_tail_to_label_pad_token_id(labels)
                    labels = labels + input_ids[context_length:]
                else:
                    input_ids = source_ids + target_ids + [self.tokenizer.eos_token_id]
                    if self.tokenizer.bos_token_id is not None:
                        input_ids = [self.tokenizer.bos_token_id] + input_ids
                        labels = [self.label_pad_token_id] + labels
                    labels = labels + target_ids + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                if len(input_ids) > self.data_args.max_input_token:
                    self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
                    input_ids = input_ids[:self.data_args.max_input_token]
                    labels = labels[:self.data_args.max_input_token]
                    attention_mask = attention_mask[:self.data_args.max_input_token]
                inputs_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)
        else:
            for prompt, answer in self.format_example(examples):
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
                if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
                    context_length = len(self.tokenizer.build_inputs_with_special_tokens(source_ids))
                    labels = [self.label_pad_token_id] * context_length + input_ids[context_length:]
                else:
                    input_ids = source_ids + target_ids + [self.tokenizer.eos_token_id]
                    context_length = len(source_ids)
                    if self.tokenizer.bos_token_id is not None:
                        input_ids = [self.tokenizer.bos_token_id] + input_ids
                        context_length = context_length + 1
                    labels = [self.label_pad_token_id] * context_length + target_ids + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                if len(input_ids) > self.data_args.max_input_token:
                    self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
                    input_ids = input_ids[:self.data_args.max_input_token]
                    labels = labels[:self.data_args.max_input_token]
                    attention_mask = attention_mask[:self.data_args.max_input_token]
                inputs_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)
        return {'input_ids': inputs_list, 'attention_mask': attention_mask_list, 'labels': labels_list}

    def preprocess_eval_supervised_fine_tuning_dataset(self, examples):
        inputs_list = []
        attention_mask_list = []
        labels_list = []
        for prompt, answer in self.format_example(examples):
            source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids)
                labels = target_ids + [self.tokenizer.eos_token_id]
            else:
                input_ids = source_ids
                if self.tokenizer.bos_token_id is not None:
                    input_ids = [self.tokenizer.bos_token_id] + source_ids
                labels = target_ids + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
                input_ids = input_ids[:self.data_args.max_input_token]
                attention_mask = attention_mask[:self.data_args.max_input_token]
            inputs_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        return {'input_ids': inputs_list, 'attention_mask': attention_mask_list, 'labels': labels_list}

    def preprocess_train_reward_model_dataset(self, examples):
        accept_list, reject_list = [], []
        for prompt, answer in self.format_example(examples):
            source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = self.tokenizer.encode(text=answer[0], add_special_tokens=False)
            reject_ids = self.tokenizer.encode(text=answer[1], add_special_tokens=False)
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                accept_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, accept_ids)
                reject_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, reject_ids)
            else:
                if self.tokenizer.bos_token_id is not None:
                    source_ids = [self.tokenizer.bos_token_id] + source_ids
                accept_ids = source_ids + accept_ids + [self.tokenizer.eos_token_id]
                reject_ids = source_ids + reject_ids + [self.tokenizer.eos_token_id]

            if len(accept_ids) > self.data_args.max_input_token or len(reject_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
                accept_ids = accept_ids[:self.data_args.max_input_token]
                reject_ids = reject_ids[:self.data_args.max_input_token]

            accept_list.append(accept_ids)
            reject_list.append(reject_ids)
        return {'accept_ids': accept_list, 'reject_ids': reject_list}

    def preprocess_train_dpo_text_dataset(self, examples):
        prompt_list, accept_list, reject_list = [], [], []
        for prompt, answer in self.format_example(examples):
            prompt_list.append(prompt)
            accept_list.append(answer[0])
            reject_list.append(answer[1])
        return {'prompt': prompt_list, 'chosen': accept_list, 'rejected': reject_list}

    def prepare_dataset(self, test=False):

        def process_dataset(process_func, dataset, shuffle=True):
            with self.training_args.main_process_first(desc='Handle dataset.'):
                if shuffle:
                    dataset = dataset.shuffle()
                dataset = dataset.map(
                    process_func,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc='Running tokenizer on dataset'
                )
                return dataset

        if not test:
            raw_datasets = self.load_datasets_from_files()
            train_dataset = raw_datasets['train']
            if self.mode == 'pretrain':
                train_dataset = process_dataset(self.preprocess_pretrain_dataset, train_dataset)
            elif self.mode == 'sft_train':
                train_dataset = process_dataset(self.preprocess_train_supervised_fine_tuning_dataset, train_dataset)
            elif self.mode == 'rm_train':
                train_dataset = process_dataset(self.preprocess_train_reward_model_dataset, train_dataset)
            elif self.mode == 'ppo_train':
                train_dataset = process_dataset(self.preprocess_eval_supervised_fine_tuning_dataset, train_dataset)
            elif self.mode == 'dpo_train':
                train_dataset = process_dataset(self.preprocess_train_dpo_text_dataset, train_dataset)
            self.logger.debug(f'Train dataset nums: {len(train_dataset)}')

            eval_dataset = None
            if self.training_args.do_eval:
                if 'validation' not in raw_datasets.keys():
                    raise ValueError('do_eval requires a validation dataset')
                eval_dataset = raw_datasets['validation']
                if self.mode == 'pretrain':
                    eval_dataset = process_dataset(self.preprocess_pretrain_dataset, eval_dataset, False)
                elif self.mode == 'sft_train':
                    eval_dataset = process_dataset(self.preprocess_eval_supervised_fine_tuning_dataset, eval_dataset, False)
                elif self.mode == 'rm_train':
                    eval_dataset = process_dataset(self.preprocess_train_reward_model_dataset, eval_dataset, False)
                elif self.mode == 'dpo_train':
                    eval_dataset = process_dataset(self.preprocess_train_dpo_text_dataset, eval_dataset, False)
                self.logger.debug(f'Validation dataset nums: {len(eval_dataset)}')
            return train_dataset, eval_dataset
        else:
            raw_datasets = self.load_datasets_from_files(test=True)
            test_dataset = raw_datasets['test']
            if self.mode == 'sft_batch_test':
                test_dataset = process_dataset(self.preprocess_eval_supervised_fine_tuning_dataset, test_dataset, False)
            elif self.mode == 'rm_batch_test':
                test_dataset = process_dataset(self.preprocess_train_reward_model_dataset, test_dataset, False)
            self.logger.debug(f'Test dataset nums: {len(test_dataset)}')
            return test_dataset


class DataCollatorForRewardModelTraining(DataCollatorWithPadding):
    def __init__(self, tokenizer, return_tensors):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors

    def __call__(self, features):
        features = [
            {'input_ids': feature[key], 'attention_mask': [1] * len(feature[key])}
            for key in ('accept_ids', 'reject_ids') for feature in features
        ]
        return super().__call__(features)
