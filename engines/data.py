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

    def load_tokenizer(self, model_path):
        if self.model_args.model_type in ['chatglm', 'baichuan', 'internlm', 'aquila', 'moss', 'qwen']:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        elif self.model_args.model_type == 'falcon':
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=self.model_args.padding_side)
        elif self.model_args.model_type == 'rwkv':
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif self.model_args.model_type == 'bloom':
            tokenizer = BloomTokenizerFast.from_pretrained(model_path)
        elif self.model_args.model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=self.model_args.use_fast_tokenizer,
                                                       padding_side=self.model_args.padding_side)
        else:
            raise

        if self.model_args.model_type == 'qwen':
            tokenizer.eos_token_id = 151643
            tokenizer.eos_token = '<|endoftext|>'
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0

        return tokenizer

    def load_datasets_from_files(self, test=False):
        data_files = {}
        if not test:
            if self.data_args.train_file_dir is not None and os.path.exists(self.data_args.train_file_dir):
                train_data_files = glob(
                    f'{self.data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                    f'{self.data_args.train_file_dir}/**/*.jsonl', recursive=True)
                self.logger.info(f"train files: {', '.join(train_data_files)}")
                data_files['train'] = train_data_files
            if self.training_args.do_eval and self.data_args.validation_file_dir is not None \
                    and os.path.exists(self.data_args.validation_file_dir):
                eval_data_files = glob(
                    f'{self.data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                    f'{self.data_args.validation_file_dir}/**/*.jsonl', recursive=True)
                self.logger.info(f"eval files: {', '.join(eval_data_files)}")
                data_files['validation'] = eval_data_files
            raw_datasets = load_dataset(
                'json',
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
            )
            if self.training_args.do_eval and 'validation' not in raw_datasets.keys() \
                    and self.data_args.dev_ratio > 0.0:
                raw_datasets['validation'] = load_dataset(
                    'json',
                    data_files=data_files,
                    split=f'train[:{self.data_args.dev_ratio}%]',
                    cache_dir=self.model_args.cache_dir,
                )
                raw_datasets['train'] = load_dataset(
                    'json',
                    data_files=data_files,
                    split=f'train[{self.data_args.dev_ratio}%:]',
                    cache_dir=self.model_args.cache_dir,
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

    def format_example(self, examples):
        for i in range(len(examples['instruction'])):
            if examples['instruction'][i] and examples['output'][i]:
                query, answer = examples['instruction'][i], examples['output'][i]
                query = query + examples['input'][i] if examples['input'][i] else query
                if 'history' in examples and (history := examples['history'][i]) is not None:
                    prompt = self.prompt_template.get_prompt(query, history)
                else:
                    prompt = self.prompt_template.get_prompt(query, [])
                yield prompt, answer

    def preprocess_train_supervised_fine_tuning_dataset(self, examples):
        # ChatGLM1: https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py#L323
        # ChatGLM2: https://huggingface.co/THUDM/chatglm2-6b/blob/main/tokenization_chatglm.py#L171
        # Baichuan: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/tokenization_baichuan.py#L152
        # internlm: https://huggingface.co/internlm/internlm-chat-7b/blob/main/tokenization_internlm.py#L179
        # moss: https://huggingface.co/fnlp/moss-moon-003-sft/blob/main/tokenization_moss.py#L226
        # Llama: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L255
        inputs_list = []
        labels_list = []
        for prompt, answer in self.format_example(examples):
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
                input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
                context_length = len(self.tokenizer.build_inputs_with_special_tokens(source_ids))
                labels = [self.label_pad_token_id] * context_length + input_ids[context_length:]
            else:
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True)
                target_ids = self.tokenizer.encode(text=answer, add_special_tokens=True)
                input_ids = source_ids + target_ids + [self.tokenizer.eos_token_id]
                context_length = len(source_ids)
                if self.tokenizer.bos_token_id is not None:
                    input_ids = [self.tokenizer.bos_token_id] + input_ids
                    context_length = context_length + 1
                labels = [self.label_pad_token_id] * context_length + target_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
            inputs_list.append(input_ids)
            labels_list.append(labels)
        return {'input_ids': inputs_list, 'labels': labels_list}

    def preprocess_eval_supervised_fine_tuning_dataset(self, examples):
        inputs_list, labels_list = [], []
        for prompt, answer in self.format_example(examples):
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                target_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
                input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids)
                labels = target_ids + [self.tokenizer.eos_token_id]
            else:
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True)
                target_ids = self.tokenizer.encode(text=answer, add_special_tokens=True)
                input_ids = source_ids
                if self.tokenizer.bos_token_id is not None:
                    input_ids = [self.tokenizer.bos_token_id] + source_ids
                labels = target_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
            inputs_list.append(input_ids)
            labels_list.append(labels)
        return {'input_ids': inputs_list, 'labels': labels_list}

    def preprocess_train_reward_model_dataset(self, examples):
        accept_list, reject_list = [], []
        for prompt, answer in self.format_example(examples):
            if self.model_args.model_type in ('chatglm', 'baichuan', 'internlm', 'moss', 'llama'):
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                accept_ids = self.tokenizer.encode(text=answer[0], add_special_tokens=False)
                reject_ids = self.tokenizer.encode(text=answer[1], add_special_tokens=False)
                accept_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, accept_ids)
                reject_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, reject_ids)
            else:
                source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True)
                accept_ids = self.tokenizer.encode(text=answer[0], add_special_tokens=True)
                reject_ids = self.tokenizer.encode(text=answer[1], add_special_tokens=True)
                if self.tokenizer.bos_token_id is not None:
                    source_ids = [self.tokenizer.bos_token_id] + source_ids
                accept_ids = source_ids + accept_ids + [self.tokenizer.eos_token_id]
                reject_ids = source_ids + reject_ids + [self.tokenizer.eos_token_id]
            if len(accept_ids) > self.data_args.max_input_token or len(reject_ids) > self.data_args.max_input_token:
                self.logger.warning(f'The token length of some sentences exceeds {self.data_args.max_input_token}.')
            accept_list.append(accept_ids)
            reject_list.append(reject_ids)
        return {'accept_ids': accept_list, 'reject_ids': reject_list}

    def prepare_dataset(self, test=False):

        def propocess_dataset(process_func, dataset, shuffle=True):
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
            if self.mode == 'sft_train':
                train_dataset = propocess_dataset(self.preprocess_train_supervised_fine_tuning_dataset, train_dataset)
            elif self.mode == 'rm_train':
                train_dataset = propocess_dataset(self.preprocess_train_reward_model_dataset, train_dataset)
            self.logger.debug(f'Train dataset nums: {len(train_dataset)}')

            eval_dataset = None
            if self.training_args.do_eval:
                if 'validation' not in raw_datasets.keys():
                    raise ValueError('do_eval requires a validation dataset')
                eval_dataset = raw_datasets['validation']
                if self.mode == 'sft_train':
                    eval_dataset = propocess_dataset(self.preprocess_eval_supervised_fine_tuning_dataset, eval_dataset, False)
                elif self.mode == 'rm_train':
                    eval_dataset = propocess_dataset(self.preprocess_train_reward_model_dataset, eval_dataset, False)
                self.logger.debug(f'Validation dataset nums: {len(eval_dataset)}')
            return train_dataset, eval_dataset
        else:
            raw_datasets = self.load_datasets_from_files(test=True)
            test_dataset = raw_datasets['test']
            if self.mode == 'rm_batch_test':
                test_dataset = propocess_dataset(self.preprocess_train_reward_model_dataset, test_dataset, False)
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
