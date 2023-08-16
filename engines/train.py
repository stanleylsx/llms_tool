# -*- coding: utf-8 -*-
# @Time : 2023/7/7 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
from engines.models import BaseModels
from engines.utils.print_parameters import print_trainable_parameters
from engines.utils.metrics import Metrics
from peft import LoraConfig, AdaLoraConfig, PromptTuningConfig, PromptEncoderConfig, PrefixTuningConfig
from peft import TaskType, get_peft_model
from engines.utils.trainer import SFTTrainer, RewardTrainer
from transformers import DataCollatorForSeq2Seq
from engines.data import DataCollatorForRewardModelTraining
from trl import AutoModelForCausalLMWithValueHead
import torch
import math


class Train(BaseModels):
    def __init__(self, data_manager, config, logger):
        super().__init__(data_manager, config, logger)
        self.data_manager = data_manager
        self.generating_args = config.generating_args
        if isinstance(self.training_args.lora_target, str):
            self.lora_target_modules = [target.strip() for target in self.training_args.lora_target.split(',')]
        assert self.training_args.fine_tuning_type in [
            'lora', 'full', 'adalora', 'prompt_tuning', 'p_tuning', 'prefix_tuning'], 'Invalid fine-tuning method.'
        self.qlore = False
        self.logger = logger
        self.metrics = Metrics(data_manager, logger)

    def construct_base_model(self, model):
        self.logger.info(f'Fine tuning type: {self.training_args.fine_tuning_type}')
        if self.training_args.fine_tuning_type == 'full':
            if self.model_args.quantization_bit is not None:
                raise ValueError('Full-parameter fine-tuning does not support quantization.')
            self.logger.info('Full-parameters training.')
            model = model.float()
            print_trainable_parameters(model, logger=self.logger)
        elif self.training_args.fine_tuning_type == 'lora':
            self.logger.info('Init new peft model.')
            if self.model_args.quantization_bit is None:
                self.logger.info('Adapter lora training.')
                self.qlore = True
            else:
                if self.model_args.quantization == 'cpm':
                    raise ValueError('Quantization CPM does not support qlora train.')
                self.logger.info('Adapter qlora training.')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.training_args.lora_rank,
                lora_alpha=self.training_args.lora_alpha,
                lora_dropout=self.training_args.lora_dropout,
                bias=self.training_args.lora_bias,
                target_modules=self.lora_target_modules,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif self.training_args.fine_tuning_type == 'adalora':
            self.logger.info('Init new peft model.')
            if self.model_args.quantization_bit is None:
                self.logger.info('Adapter adalora training.')
                self.qlore = True
            else:
                if self.model_args.quantization == 'cpm':
                    raise ValueError('Quantization CPM does not support qlora train.')
                self.logger.info('Adapter qadalora training.')
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.training_args.lora_rank,
                lora_alpha=self.training_args.lora_alpha,
                lora_dropout=self.training_args.lora_dropout,
                init_r=self.training_args.adalora_init_r,
                beta1=self.training_args.adalora_beta,
                beta2=self.training_args.adalora_beta,
                tinit=self.training_args.adalora_tinit,
                tfinal=self.training_args.adalora_tfinal,
                deltaT=self.training_args.adalora_delta_t,
                target_modules=self.lora_target_modules,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif self.training_args.fine_tuning_type == 'prompt_tuning':
            self.logger.info('Init new peft model.')
            self.logger.info('Adapter prompt tuning training.')
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.training_args.num_virtual_tokens,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif self.training_args.fine_tuning_type == 'p_tuning':
            self.logger.info('Init new peft model.')
            self.logger.info('Adapter P-tuning training.')
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.training_args.num_virtual_tokens,
                encoder_hidden_size=self.training_args.prompt_encoder_hidden_size
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif self.training_args.fine_tuning_type == 'prefix_tuning':
            self.logger.info('Init new peft model.')
            self.logger.info('Adapter prefix tuning training.')
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.training_args.num_virtual_tokens,
                encoder_hidden_size=self.training_args.prompt_encoder_hidden_size,
                prefix_projection=True,
            )
            model.gradient_checkpointing_disable()
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        return model

    def set_train_enviroment(self, model):
        if self.training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        else:
            model.config.use_cache = True

        try:
            model.enable_input_require_grads()
        except:
            self.logger.warning('Could not enable input require_grads on model, skipping.')

        if torch.cuda.device_count() > 1:
            # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    def supervised_fine_tuning(self):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        model = self.construct_base_model(model)
        self.logger.info(f'Model struct:\n{model}')
        self.set_train_enviroment(model)

        train_dataset, eval_dataset = self.data_manager.prepare_dataset()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            return_tensors='pt',
            label_pad_token_id=self.data_manager.label_pad_token_id,
        )
        trainer = SFTTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        self.logger.info('*** Start training. ***')
        checkpoint = None
        if self.training_args.resume_from_checkpoint:
            checkpoint = self.training_args.output_dir
            self.logger.info(f'Resume checkpoint from {checkpoint}')
        try:
            trainer_result = trainer.train(resume_from_checkpoint=checkpoint)
        except ValueError:
            self.logger.warning(f"Can't find a valid checkpoint at {checkpoint}")
            trainer_result = trainer.train()
        metrics = trainer_result.metrics
        self.logger.info(f'Training metrics: {metrics}')
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        self.logger.info(f'Saving model checkpoint to {self.training_args.output_dir}')
        trainer.save_state()
        trainer.save_model()

        if self.training_args.do_eval and eval_dataset:
            self.logger.info('*** Start evaluating. ***')
            gen_kwargs = self.generating_args.to_dict()
            metrics = trainer.evaluate(eval_dataset=eval_dataset, **gen_kwargs)
            try:
                perplexity = math.exp(metrics['eval_loss'])
            except OverflowError:
                perplexity = float('inf')
            metrics['perplexity'] = perplexity
            self.logger.info(f'Evaluating metrics: {metrics}')
            trainer.log_metrics('eval', metrics)
            trainer.save_metrics('eval', metrics)

    def train_reward_model(self, test=False):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        reward_model = self.load_reward_model(model, vhead_dir=self.training_args.output_dir)
        if test and not self.has_vhead:
            raise Exception('Reward model is not correctly loaded.')
        if not self.has_vhead:
            model = self.construct_base_model(model)
            if self.model_args.model_type == 'chatglm' and any(
                    key.endswith('rotary_pos_emb') for key, _ in model.named_modules()):
                model.lm_head = model.transformer.output_layer
            reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        self.logger.info(f'Model struct:\n{reward_model}')
        print_trainable_parameters(reward_model, self.logger)
        self.set_train_enviroment(model)
        data_collator = DataCollatorForRewardModelTraining(tokenizer=self.tokenizer, return_tensors='pt')
        if not test:
            train_dataset, eval_dataset = self.data_manager.prepare_dataset()
            self.training_args.remove_unused_columns = False
            trainer = RewardTrainer(
                model_type=self.model_args.model_type,
                model=reward_model,
                args=self.training_args,
                train_dataset=train_dataset if self.training_args.do_train else None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.metrics.computer_training_reward_metric
            )
            train_result = trainer.train()
            metrics = train_result.metrics
            self.logger.info(f'Training metrics: {metrics}')
            trainer.log_metrics('train', metrics)
            trainer.save_metrics('train', metrics)
            self.logger.info(f'Saving model checkpoint to {self.training_args.output_dir}')
            trainer.save_state()
            trainer.save_model()

            if self.training_args.do_eval and eval_dataset:
                self.logger.info('*** Start evaluating. ***')
                metrics = trainer.evaluate(eval_dataset)
                self.logger.info(f'Evaluating metrics: {metrics}')
                trainer.log_metrics('eval', metrics)
                trainer.save_metrics('eval', metrics)
        else:
            test_dataset = self.data_manager.prepare_dataset(test=True)
            self.training_args.remove_unused_columns = False
            trainer = RewardTrainer(
                model_type=self.model_args.model_type,
                model=reward_model,
                args=self.training_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.metrics.computer_training_reward_metric
            )
            metrics = trainer.evaluate(test_dataset, metric_key_prefix='test')
            self.logger.info(f'Test metrics: {metrics}')
            trainer.log_metrics('test', metrics)
            trainer.save_metrics('test', metrics)

    def train_ppo(self):
        pass
