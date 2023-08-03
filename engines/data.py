# -*- coding: utf-8 -*-
# @Time : 2023/7/10 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
from transformers import AutoTokenizer, LlamaTokenizer, BloomTokenizerFast
from engines.utils.prompt_template import Template
from datasets import load_dataset
from glob import glob
import os


class DataManager:
    def __init__(self, config, logger):
        self.logger = logger
        self.data_args = config.data_args
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.prompt_template = Template(self.data_args.prompt_template)
        logger.info(f'Load tokenizer: {self.model_args.model_path}')
        self.tokenizer = self.load_tokenizer(self.model_args.model_path)
        self.logger.info(f'Tokenizer: {self.tokenizer}')
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

        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({
                'eos_token': '</s>',
                'bos_token': '<sop>',
                'unk_token': '<unk>',
            })
        if self.model_args.model_type == 'qwen':
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_datasets(self):
        data_files = {}
        if self.data_args.train_file_dir is not None and os.path.exists(self.data_args.train_file_dir):
            train_data_files = glob(
                f'{self.data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{self.data_args.train_file_dir}/**/*.jsonl', recursive=True)
            self.logger.info(f"train files: {', '.join(train_data_files)}")
            data_files['train'] = train_data_files
        if self.training_args.do_predict and self.data_args.validation_file_dir is not None \
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
        if self.training_args.do_predict and 'validation' not in raw_datasets.keys() \
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
        # Llama: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L253
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

    def prepare_supervised_fine_tuning_dataset(self):
        raw_datasets = self.load_datasets()
        train_dataset = raw_datasets['train']
        with self.training_args.main_process_first(desc='Handle validation dataset.'):
            train_dataset = train_dataset.shuffle().map(
                self.preprocess_train_supervised_fine_tuning_dataset,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc='Running tokenizer on train dataset'
            )
            self.logger.debug(f'Train dataset nums: {len(train_dataset)}')

        eval_dataset = None
        if self.training_args.do_eval:
            if 'validation' not in raw_datasets.keys():
                raise ValueError('do_eval requires a validation dataset')
            eval_dataset = raw_datasets['validation']

            with self.training_args.main_process_first(desc='Handle validation dataset.'):
                eval_dataset = eval_dataset.map(
                    self.preprocess_eval_supervised_fine_tuning_dataset,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=eval_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc='Running tokenizer on validation dataset'
                )
                self.logger.debug(f'Validation dataset nums: {len(eval_dataset)}')
        return train_dataset, eval_dataset
