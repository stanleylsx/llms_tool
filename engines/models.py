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
from trl import AutoModelForCausalLMWithValueHead
from accelerate import infer_auto_device_map, dispatch_model
from engines.utils.glm_multi_gpus import auto_configure_device_map
from engines.utils.print_parameters import summary
from engines.utils.cpm_quantizer import QuantizedLinear
from peft.utils import CONFIG_NAME, WEIGHTS_NAME
from peft import PeftModel
from types import MethodType
import os
import torch


class BaseModels:
    def __init__(self, data_manager, config, logger):
        self.logger = logger
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.mode = config.mode
        self.tokenizer = data_manager.tokenizer
        self.data_manager = data_manager
        self.has_peft = False
        self.has_vhead = False

    def load_adapter(self, model, adapter_dir):
        if adapter_dir is None:
            return model
        if os.path.exists(os.path.join(adapter_dir, WEIGHTS_NAME)) and os.path.exists(os.path.join(adapter_dir, CONFIG_NAME)):
            self.logger.info(f'Found adapter model at {adapter_dir} and load it.')
            self.has_peft = True
            model = PeftModel.from_pretrained(model, adapter_dir)
            if self.mode in ('web_inference', 'merge_peft_model', 'save_quantized_model'):
                self.logger.info('Merge peft model.')
                model = model.merge_and_unload()
        else:
            self.logger.info(f'The given dir: {adapter_dir} may be not have adapter checkpoint.')
        return model

    def load_reward_model(self, model, vhead_dir):
        if os.path.exists(vhead_path := os.path.join(vhead_dir, 'vhead.bin')):
            self.logger.info(f'Found v_head model at {vhead_dir} and load it.')
            model = self.load_adapter(model, adapter_dir=vhead_dir)
            self.has_vhead = True
            if self.model_args.model_type == 'chatglm' and any(
                    key.endswith('rotary_pos_emb') for key, _ in model.named_modules()):
                model.lm_head = model.transformer.output_layer
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
            model.load_state_dict(torch.load(vhead_path), strict=False)
        else:
            self.logger.info(f'The given dir: {vhead_dir} may be not have v_head checkpoint.')
        return model

    def load_base_model(self):
        config_kwargs = {'cache_dir': self.model_args.cache_dir,
                         'torch_dtype': self.model_args.torch_dtype}
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
                config_kwargs['device_map'] = 'auto'
                self.logger.info('Quantifying(bnb) model to {} bit.'.format(self.model_args.quantization_bit))
            elif self.model_args.quantization == 'cpm':
                self.logger.info('Quantifying(cpm) model to {} bit.'.format(self.model_args.quantization_bit))
        else:
            if self.model_args.model_type != 'chatglm':
                config_kwargs['device_map'] = 'auto'

        if self.model_args.checkpoint_dir is not None and self.training_args.fine_tuning_type == 'full':
            model_to_load = self.model_args.checkpoint_dir
        else:
            model_to_load = self.model_args.model_path

        if self.model_args.model_type == 'chatglm':
            model = AutoModel.from_pretrained(model_to_load, trust_remote_code=True, **config_kwargs)
        elif self.model_args.model_type in ['falcon', 'baichuan', 'aquila', 'internlm', 'moss', 'qwen']:
            model = AutoModelForCausalLM.from_pretrained(model_to_load, trust_remote_code=True, **config_kwargs)
            if self.model_args.model_type == 'qwen':
                model.generate = MethodType(PreTrainedModel.generate, model)
        elif self.model_args.model_type == 'rwkv':
            model = RwkvForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        elif self.model_args.model_type == 'llama':
            model = LlamaForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        elif self.model_args.model_type == 'bloom':
            model = BloomForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        else:
            raise

        if self.model_args.quantization_bit is not None and self.model_args.quantization == 'cpm':
            model = self.quantize(model, self.model_args.quantization_bit)
            model.tie_weights()
            if self.model_args.model_type != 'chatglm':
                device_map = infer_auto_device_map(model)
            else:
                device_map = auto_configure_device_map(torch.cuda.device_count(), model)
            model = dispatch_model(model, device_map=device_map)
        else:
            if self.model_args.model_type == 'chatglm':
                model.tie_weights()
                device_map = auto_configure_device_map(torch.cuda.device_count(), model)
                model = dispatch_model(model, device_map=device_map)

        if os.path.exists(model_to_load + '/generation_config.json'):
            model.generation_config = GenerationConfig.from_pretrained(model_to_load)
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
                setattr(super_module, key.split('.')[-1], quantized_liner)
        return model

    def save_quantized_model(self):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        model = self.load_adapter(model, adapter_dir=self.model_args.checkpoint_dir)
        self.logger.info('Saving quantized model.')
        model.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.tokenizer.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.logger.info(f'Quantize done, model saved to {self.model_args.quantized_or_merged_output_dir}')

    def merge_peft_model(self):
        if self.model_args.checkpoint_dir is None:
            self.logger.error(f'checkpoint_dir is None.')
        if not os.path.exists(os.path.join(self.model_args.checkpoint_dir, WEIGHTS_NAME)) \
                and os.path.exists(os.path.join(self.model_args.checkpoint_dir, CONFIG_NAME)):
            self.logger.error(f'Peft checkpoint not found at {self.model_args.checkpoint_dir}.')
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        model = self.load_adapter(model, adapter_dir=self.model_args.checkpoint_dir)
        if not self.has_peft:
            self.logger.error('Peft checkpoint not found.')
        self.logger.info(f'Base model: {self.model_args.model_type}')
        self.logger.info(f'Peft model: {self.model_args.checkpoint_dir}')
        self.logger.info('Loading LoRA for causal language model')
        tokenizer = self.data_manager.load_tokenizer(self.model_args.checkpoint_dir)
        self.logger.info('Saving to Hugging Face format...')
        tokenizer.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        model.save_pretrained(self.model_args.quantized_or_merged_output_dir)
        self.logger.info(f'Merge done, model saved to {self.model_args.quantized_or_merged_output_dir}')

    def show_model_info(self):
        self.logger.info(f'Load base model from {self.model_args.model_path}')
        model = self.load_base_model()
        model = self.load_adapter(model, adapter_dir=self.model_args.checkpoint_dir)
        info = summary(model, max_level=3)
        self.logger.info(f'Model struct:\n{model}')
        self.logger.info(f'Model parameter:\n{info}')
