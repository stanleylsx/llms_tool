# -*- coding: utf-8 -*-
# @Time : 2023/7/7 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
from engines.models import BaseModels
from engines.utils.print_parameters import print_trainable_parameters
from engines.utils.metrics import Metrics
from engines.data import DataCollatorForRewardModelTraining
from engines.utils.trainer import SFTTrainer, RewardTrainer, MyPPOTrainer
from peft import LoraConfig, AdaLoraConfig, PromptTuningConfig, PromptEncoderConfig, PrefixTuningConfig
from peft import TaskType, get_peft_model
from copy import deepcopy
from transformers import DataCollatorForSeq2Seq
from config import TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, set_seed, DPOTrainer
from tqdm import tqdm
import torch
import math
import os


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
            if str(model.dtype) in ['torch.float16', 'torch.bfloat16'] and (
                    self.training_args.fp16 or self.training_args.bf16):
                self.logger.warning('If you need training full model with fp16 or bf16,'
                                    ' you should load model with fp32(set torch_dtype=float32)')
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

    def set_train_environment(self, model):
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

    def supervised_fine_tuning(self, test=False):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            return_tensors='pt',
            label_pad_token_id=self.data_manager.label_pad_token_id,
        )
        if not test:
            model = self.construct_base_model(model)
            self.set_train_environment(model)
            self.logger.info(f'Model struct:\n{model}')
            train_dataset, eval_dataset = self.data_manager.prepare_dataset()
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
                self.logger.info(f'Evaluating metrics: {metrics}')
                trainer.log_metrics('eval', metrics)
                trainer.save_metrics('eval', metrics)
        else:
            model = self.load_adapter(model, adapter_dir=self.training_args.output_dir)
            model.eval()
            self.logger.info(f'Model struct:\n{model}')
            test_dataset = self.data_manager.prepare_dataset(test=True)
            trainer = SFTTrainer(
                model=model,
                args=self.training_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.metrics.computer_supervised_fine_tuning_metric
            )
            gen_kwargs = self.generating_args.to_dict()
            self.logger.info('*** Start testing. ***')
            test_results = trainer.predict(test_dataset, metric_key_prefix='test', **gen_kwargs)
            metrics = test_results.metrics
            perplexity = math.exp(metrics['test_loss'])
            metrics['perplexity'] = perplexity
            self.logger.info(f'Test metrics: {metrics}')
            trainer.log_metrics('test', metrics)
            trainer.save_metrics('test', metrics)

    def train_reward_model(self, test=False):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        reward_model = self.load_reward_model(model, vhead_dir=self.training_args.output_dir)
        if test and not self.has_vhead:
            raise Exception('Reward model is not correctly loaded.')
        self.set_train_environment(reward_model)
        if not self.has_vhead:
            model = self.construct_base_model(model)
            if self.model_args.model_type == 'chatglm' and any(
                    key.endswith('rotary_pos_emb') for key, _ in model.named_modules()):
                model.lm_head = model.transformer.output_layer
            reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        self.logger.info(f'Model struct:\n{reward_model}')
        data_collator = DataCollatorForRewardModelTraining(tokenizer=self.tokenizer, return_tensors='pt')
        if not test:
            print_trainable_parameters(reward_model, self.logger)
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
            reward_model.eval()
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
            self.logger.info('*** Start testing. ***')
            test_results = trainer.predict(test_dataset, metric_key_prefix='test')
            metrics = test_results.metrics
            self.logger.info(f'Test metrics: {metrics}')
            trainer.log_metrics('test', metrics)
            trainer.save_metrics('test', metrics)

    def train_ppo(self):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        reward_model = self.load_reward_model(model, vhead_dir=self.model_args.reward_model_checkpoint)
        reward_model.eval()
        self.logger.info(f'Reward model struct:\n{reward_model}')

        sft_model = self.load_adapter(model, adapter_dir=self.model_args.checkpoint_dir)

        if self.model_args.model_type == 'chatglm' and any(
                key.endswith('rotary_pos_emb') for key, _ in sft_model.named_modules()):
            sft_model.lm_head = sft_model.transformer.output_layer

        ppo_model = self.construct_base_model(sft_model)
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_model)
        self.set_train_environment(ppo_model)
        self.logger.info(f'PPO model struct:\n{ppo_model}')
        print_trainable_parameters(ppo_model, logger=self.logger)

        train_dataset, _ = self.data_manager.prepare_dataset()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            return_tensors='pt',
            label_pad_token_id=self.tokenizer.pad_token_id
        )
        output_dir = self.training_args.output_dir
        config = PPOConfig(
            steps=self.training_args.ppo_steps,
            learning_rate=self.training_args.learning_rate,
            batch_size=self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps,
            mini_batch_size=self.training_args.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            target_kl=self.training_args.target_kl,
            ppo_epochs=self.training_args.ppo_epochs,
            seed=self.training_args.seed,
            init_kl_coef=self.training_args.init_kl_coef,
            adap_kl_ctrl=self.training_args.adap_kl_ctrl,
            project_kwargs={'logging_dir': output_dir}
        )
        set_seed(config.seed)
        ppo_trainer = MyPPOTrainer(
            model_type=self.model_args.model_type,
            config=config,
            model=ppo_model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=train_dataset,
            data_collator=data_collator
        )
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs = self.data_manager.generating_args_preprocess(gen_kwargs)

        total_steps = config.total_ppo_epochs
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            if step >= total_steps:
                break

            queries = batch['input_ids']
            batch['query'] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
            queries = [i.squeeze(0) for i in queries]

            responses = ppo_trainer.generate(queries, return_prompt=False, **gen_kwargs)
            batch['response'] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            reward_input_batch = ppo_trainer.prepare_model_inputs(queries, responses)
            _, _, values = reward_model(**reward_input_batch, output_hidden_states=True, return_dict=True)

            values = torch.transpose(values, 1, 0) if self.model_args.model_type == 'chatglm' else values
            scores = [reward for reward in values[:, -1].detach().cpu()]

            unpad_queries = []
            for query in queries:
                query = query.detach().cpu().numpy()
                query = query[query != self.tokenizer.pad_token_id]
                unpad_queries.append(torch.tensor(query))

            responses = [i.squeeze(0) for i in responses]
            stats = ppo_trainer.step(queries, responses, scores)
            ppo_trainer.log_stats(stats, batch, scores)
            self.logger.debug(f'Step {step}/{total_steps}: reward score:{scores}')
            if (step + 1) % self.training_args.save_steps == 0:
                ppo_trainer.save_pretrained(os.path.join(self.training_args.output_dir, f'checkpoint-{step + 1}'))
        ppo_trainer.save_pretrained(self.training_args.output_dir)

    def train_dpo(self):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        model = self.load_adapter(model, adapter_dir=self.model_args.checkpoint_dir)
        ref_model = deepcopy(model)
        model = self.construct_base_model(model)
        self.set_train_environment(model)
        self.logger.info(f'Model struct:\n{model}')
        train_dataset, eval_dataset = self.data_manager.prepare_dataset()
        training_args = self.training_args.to_dict()
        training_args |= {'remove_unused_columns': False}
        training_args = TrainingArguments(**training_args)
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_length=self.data_manager.data_args.max_input_token
        )
        self.logger.info('*** Start training. ***')
        checkpoint = None
        if self.training_args.resume_from_checkpoint:
            checkpoint = self.training_args.output_dir
            self.logger.info(f'Resume checkpoint from {checkpoint}')
        try:
            trainer_result = dpo_trainer.train(resume_from_checkpoint=checkpoint)
        except ValueError:
            self.logger.warning(f"Can't find a valid checkpoint at {checkpoint}")
            trainer_result = dpo_trainer.train()
        metrics = trainer_result.metrics
        self.logger.info(f'Training metrics: {metrics}')
        dpo_trainer.log_metrics('train', metrics)
        dpo_trainer.save_metrics('train', metrics)
        self.logger.info(f'Saving model checkpoint to {self.training_args.output_dir}')
        dpo_trainer.save_state()
        dpo_trainer.save_model()

        if self.training_args.do_eval and eval_dataset:
            self.logger.info('*** Start evaluating. ***')
            metrics = dpo_trainer.evaluate()
            self.logger.info(f'Evaluating metrics: {metrics}')
            dpo_trainer.log_metrics('eval', metrics)
            dpo_trainer.save_metrics('eval', metrics)
