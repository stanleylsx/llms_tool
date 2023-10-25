# -*- coding: utf-8 -*-
# @Time : 2023/7/25 22:35
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : check_load_config.py
# @Software: PyCharm
from transformers import HfArgumentParser
from config import WorkingMode, ModelArguments, DataTrainingArguments, TrainingArguments, GeneratingArguments
import os


class Configure:
    def __init__(self):
        parser = HfArgumentParser((WorkingMode, ModelArguments, DataTrainingArguments, TrainingArguments, GeneratingArguments))
        self.mode, self.model_args, self.data_args, self.training_args, self.generating_args = parser.parse_args_into_dataclasses()
        self.mode = self.mode.mode

        assert self.model_args.quantization_bit is None or self.training_args.fine_tuning_type in (
            'lora', 'adalora'), 'Quantization is only compatible with the LoRA method(QLora).'

        if self.data_args.prompt_template == 'default':
            print('Please specify `prompt_template` if you are using other pre-trained models.')

        if self.training_args.do_train:
            print(
                f'Process rank: {self.training_args.local_rank}\n'
                f'device: {self.training_args.device}\n'
                f'n_gpu: {self.training_args.n_gpu}\n'
                f'distributed training: {bool(self.training_args.local_rank != -1)}\n'
                f'16-bits training: {self.training_args.fp16}\n'
            )
        self.fold_check()

    def fold_check(self):
        if not os.path.exists(self.data_args.train_file_dir):
            raise ValueError('Train dataset not found.')
        if not os.path.exists(self.training_args.output_dir):
            print('Creating output_dir fold.')
            os.makedirs(self.training_args.output_dir)

        if not os.path.exists('./logs'):
            print('Creating log fold.')
            os.mkdir('./logs')
