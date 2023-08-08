# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
from transformers import AutoModel, LlamaForCausalLM, BloomForCausalLM, AutoModelForCausalLM, RwkvForCausalLM
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationConfig
from accelerate import infer_auto_device_map, dispatch_model
from engines.utils.glm_multi_gpus import auto_configure_device_map
from engines.utils.print_parameters import summary
from engines.utils.cpm_quantizer import QuantizedLinear
from peft.utils import CONFIG_NAME, WEIGHTS_NAME
from peft import PeftModel
from types import MethodType
import os
import re
import torch


class BaseModels:
    def __init__(self, data_manager, config, logger):
        self.logger = logger
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.tokenizer = data_manager.tokenizer
        self.data_manager = data_manager
        logger.info(f'Load model: {self.model_args.model_path}')
        self.model = self.load_model()
        if self.model_args.checkpoint_dir is None:
            logger.warning('Checkpoint is not found, load the original model.')
        else:
            logger.info(f'Load adapter model: {self.model_args.checkpoint_dir} ')
            self.model = self.load_adapter(self.model)
        if self.model_args.checkpoint_dir is not None:
            logger.info('Loaded fine-tuned model from checkpoint: {}'.format(self.model_args.checkpoint_dir))

    def load_model(self):
        config_kwargs = {'cache_dir': self.model_args.cache_dir}
        if self.model_args.quantization_bit is not None:
            if self.model_args.quantization == 'bnb':
                if self.model_args.quantization_bit == 8:
                    config_kwargs['load_in_8bit'] = True
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0)
                elif self.model_args.quantization_bit == 4:
                    config_kwargs['load_in_4bit'] = True
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.model_args.torch_dtype,
                        bnb_4bit_use_double_quant=self.model_args.double_quantization,
                        bnb_4bit_quant_type=self.model_args.quantization_type)
                self.logger.info('Quantifying(bnb) model to {} bit.'.format(self.model_args.quantization_bit))
            elif self.model_args.quantization == 'cpm':
                self.logger.info('Quantifying(cpm) model to {} bit.'.format(self.model_args.quantization_bit))

        if self.model_args.checkpoint_dir is not None and self.training_args.fine_tuning_type == 'full':
            model_to_load = self.model_args.checkpoint_dir
        else:
            model_to_load = self.model_args.model_path

        if self.model_args.model_type == 'chatglm':
            if self.model_args.quantization_bit is not None and self.model_args.quantization == 'cpm':
                model = AutoModel.from_pretrained(
                    model_to_load,
                    trust_remote_code=True,
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs
                ).quantize(self.model_args.quantization_bit)
            else:
                model = AutoModel.from_pretrained(
                    model_to_load,
                    trust_remote_code=True,
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs
                )
            model.tie_weights()
            device_map = auto_configure_device_map(torch.cuda.device_count(), model)
            model = dispatch_model(model, device_map=device_map)
        elif self.model_args.model_type in ['falcon', 'baichuan', 'aquila', 'internlm', 'moss', 'qwen']:
            if self.model_args.quantization_bit is not None and self.model_args.quantization == 'cpm':
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    trust_remote_code=True,
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs
                )
                model = self.quantize(model, self.model_args.quantization_bit)
                model.tie_weights()
                device_map = infer_auto_device_map(model)
                self.logger.info(device_map)
                model = dispatch_model(model, device_map=device_map)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    device_map='auto',
                    trust_remote_code=True,
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs)
            if self.model_args.model_type == 'qwen':
                model.generate = MethodType(PreTrainedModel.generate, model)
        elif self.model_args.model_type == 'rwkv':
            model = RwkvForCausalLM.from_pretrained(model_to_load, device_map='auto')
        elif self.model_args.model_type == 'llama':
            if self.model_args.quantization_bit is not None and self.model_args.quantization == 'cpm':
                model = LlamaForCausalLM.from_pretrained(
                    model_to_load,
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs)
                model = self.quantize(model, self.model_args.quantization_bit)
                model.tie_weights()
                device_map = infer_auto_device_map(model)
                self.logger.info(device_map)
                model = dispatch_model(model, device_map=device_map)
            else:
                model = LlamaForCausalLM.from_pretrained(
                    model_to_load,
                    device_map='auto',
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs)
        elif self.model_args.model_type == 'bloom':
            if self.model_args.quantization_bit is not None and self.model_args.quantization == 'cpm':
                model = BloomForCausalLM.from_pretrained(
                    model_to_load,
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs)
                model = self.quantize(model, self.model_args.quantization_bit)
                model.tie_weights()
                device_map = infer_auto_device_map(model)
                self.logger.info(device_map)
                model = dispatch_model(model, device_map=device_map)
            else:
                model = BloomForCausalLM.from_pretrained(
                    model_to_load,
                    device_map='auto',
                    torch_dtype=self.model_args.torch_dtype,
                    **config_kwargs)
        else:
            raise

        if os.path.exists(model_to_load + '/generation_config.json'):
            model.generation_config = GenerationConfig.from_pretrained(model_to_load)
        return model

    def load_adapter(self, model):
        if self.training_args.fine_tuning_type in (
                'lora', 'full', 'adalora', 'prompt_tuning', 'p_tuning', 'prefix_tuning'):
            assert os.path.exists(os.path.join(self.model_args.checkpoint_dir, WEIGHTS_NAME)), \
                'Provided path ({}) does not contain a adapter weight.'.format(self.model_args.checkpoint_dir)
            assert os.path.exists(os.path.join(self.model_args.checkpoint_dir, CONFIG_NAME)), \
                'The given checkpoint may be not a adapter checkpoint, ' \
                'please specify `--fine_tuning_type full` instead.'
            model = PeftModel.from_pretrained(model, self.model_args.checkpoint_dir)
            model = model.merge_and_unload()
        return model

    @staticmethod
    def get_module_by_name(model, module_name):
        name_list = module_name.split('.')
        for name in name_list[:-1]:
            if hasattr(model, name):
                model = getattr(model, name)
            else:
                return None, None
        if hasattr(model, name_list[-1]):
            leaf_module = getattr(model, name_list[-1])
            return model, leaf_module
        else:
            return None, None

    def quantize(self, model, bits, device=None):
        for key, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if key in ('lm_head', 'embed_out', 'output_layer'):
                    continue
                super_module, leaf_module = self.get_module_by_name(model, key)
                quantized_liner = QuantizedLinear(
                    weight_bit_width=bits,
                    weight=leaf_module.weight.to(torch.cuda.current_device()),
                    bias=leaf_module.bias,
                    dtype=leaf_module.weight.dtype,
                    device=leaf_module.weight.device if device is None else device,
                )
                setattr(super_module, key[0].split('.')[-1], quantized_liner)
        return model

    def save_quantized_model(self):
        self.logger.info('Saving quantized model.')
        self.model.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.tokenizer.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.logger.info(f'Quantize done, model saved to {self.model_args.quantized_or_merged_output_dir}')

    def merge_peft_model(self):
        self.logger.info(f'Base model: {self.model_args.model_type}')
        self.logger.info(f'Peft model: {self.model_args.checkpoint_dir}')
        self.logger.info('Loading LoRA for causal language model')
        tokenizer = self.data_manager.load_tokenizer(self.model_args.checkpoint_dir)
        base_model_token_size = self.model.get_input_embeddings().weight.size(0)
        if base_model_token_size != len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer))
            self.logger.info(f'Resize vocabulary size {base_model_token_size} to {len(tokenizer)}')
        self.logger.info('Saving to Hugging Face format...')
        tokenizer.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.model.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.logger.info(f'Merge done, model saved to {self.model_args.quantized_or_merged_output_dir}')

    def show_model_info(self):
        info = summary(self.model, max_level=3)
        self.logger.info(f'Model struct:\n{self.model}')
        self.logger.info(f'Model parameter:\n{info}')
