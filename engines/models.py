# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
from transformers import AutoModel, LlamaForCausalLM, BloomForCausalLM, AutoModelForCausalLM, RwkvForCausalLM, FalconForCausalLM
from transformers import MistralForCausalLM
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from engines.utils.dispatch_to_multi_gpu import dispatch
from engines.utils.print_parameters import summary
from engines.utils.cpm_quantizer import QuantizedLinear
from peft.utils import CONFIG_NAME, WEIGHTS_NAME
from peft import PeftModel
from types import MethodType
import os
import math
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
        self.is_deepspeed_train = False
        if world_size := os.environ.get('WORLD_SIZE') is not None:
            self.world_size = int(world_size)
            self.is_deepspeed_train = True

    def load_adapter(self, model, adapter_dir):
        if adapter_dir is None:
            return model
        if os.path.exists(os.path.join(adapter_dir, WEIGHTS_NAME)) and os.path.exists(os.path.join(adapter_dir, CONFIG_NAME)):
            self.logger.info(f'Found adapter model at {adapter_dir} and load it.')
            self.has_peft = True
            model = PeftModel.from_pretrained(model, adapter_dir)
            if self.training_args.fine_tuning_type in ('lora', 'adalora'):
                # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#L790
                if self.mode in ('merge_peft_model', 'save_quantized_model', 'ppo_train', 'dpo_train'):
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

    def use_ntk_to_expend_input_token_length(self, model):
        ntk_type = self.model_args.use_ntk
        max_input_token = self.data_manager.data_args.max_input_token
        if isinstance(model, LlamaForCausalLM):
            if max_input_token > (max_position_embeddings := getattr(model.config, 'max_position_embeddings', None)):
                factor = math.ceil(max_input_token / max_position_embeddings)
                match ntk_type:
                    case 'dynamic':
                        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L147
                        model.config.rope_scaling = {'type': 'dynamic', 'factor': factor}
                    case 'linear':
                        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L128
                        model.config.rope_scaling = {'type': 'linear', 'factor': factor}
            else:
                self.logger.warning('Current model support the length you set.')
            return model
        match self.model_args.model_type:
            case 'chatglm':
                if ntk_type == 'linear':
                    if (rope_ratio := getattr(model.config, 'rope_ratio', None)) is not None:
                        if (set_rope_ratio := math.ceil(max_input_token / 2048)) > rope_ratio:
                            # https://huggingface.co/THUDM/chatglm2-6b-32k/blob/main/modeling_chatglm.py#L141
                            model.config.rope_ratio = set_rope_ratio
                        else:
                            self.logger.warning('Current model support the length you set.')
                    else:
                        self.logger.warning('Only chatglm2-6b-32k support expend input token length.')
                else:
                    self.logger.warning('Native ChatGLM can not support dynamic NTK.')
            case 'qwen':
                if ntk_type == 'dynamic':
                    # https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/modeling_qwen.py#L1165
                    model.config.use_dynamic_ntk = True
                else:
                    self.logger.warning('Native Qwen can not support linear NTK.')
            case 'falcon':
                if model.config.alibi:
                    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/configuration_falcon.py#L181
                    self.logger.warning('`rope_scaling` is not supported when `alibi` is `True`.')
                else:
                    if max_input_token > 2048:
                        factor = math.ceil(max_input_token / 2048)
                        match ntk_type:
                            case 'dynamic':
                                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py#L143
                                model.config.rope_scaling = {'type': 'dynamic', 'factor': factor}
                            case 'linear':
                                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py#L117
                                model.config.rope_scaling = {'type': 'linear', 'factor': factor}
                    else:
                        self.logger.warning('Current model support the length you set.')
        return model

    def load_base_model(self):
        config_kwargs = {'cache_dir': self.model_args.cache_dir,
                         'torch_dtype': self.model_args.torch_dtype}
        dispatched = False
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
                if self.is_deepspeed_train:
                    device_map = {'': int(os.environ['LOCAL_RANK'])}
                    config_kwargs['device_map'] = device_map
                else:
                    config_kwargs['device_map'] = 'auto'
                self.logger.info('Quantifying(bnb) model to {} bit.'.format(self.model_args.quantization_bit))
            elif self.model_args.quantization == 'cpm':
                self.logger.info('Quantifying(cpm) model to {} bit.'.format(self.model_args.quantization_bit))
        else:
            if not self.is_deepspeed_train and self.model_args.model_type != 'chatglm':
                config_kwargs['device_map'] = 'auto'
                dispatched = True

        if self.model_args.checkpoint_dir is not None and self.training_args.fine_tuning_type == 'full':
            model_to_load = self.model_args.checkpoint_dir
        else:
            model_to_load = self.model_args.model_path

        if self.model_args.use_flash_attn:
            config_kwargs['use_flash_attention_2'] = True

        if self.model_args.model_type == 'chatglm':
            model = AutoModel.from_pretrained(model_to_load, trust_remote_code=True, **config_kwargs)
        elif self.model_args.model_type == 'falcon':
            model = FalconForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        elif self.model_args.model_type == 'mistral':
            model = MistralForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        elif self.model_args.model_type in ['baichuan', 'aquila', 'internlm', 'moss', 'xverse']:
            model = AutoModelForCausalLM.from_pretrained(model_to_load, trust_remote_code=True, **config_kwargs)
        elif self.model_args.model_type == 'qwen':
            match self.model_args.torch_dtype:
                case torch.float16:
                    config_kwargs['fp16'] = True
                case torch.bfloat16:
                    config_kwargs['bf16'] = True
                case torch.float32:
                    config_kwargs['fp32'] = True
            model = AutoModelForCausalLM.from_pretrained(model_to_load, trust_remote_code=True, **config_kwargs)
            model.generate = MethodType(PreTrainedModel.generate, model)
        elif self.model_args.model_type == 'rwkv':
            model = RwkvForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        elif self.model_args.model_type == 'llama':
            model = LlamaForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        elif self.model_args.model_type == 'bloom':
            model = BloomForCausalLM.from_pretrained(model_to_load, **config_kwargs)
        else:
            raise

        if self.model_args.use_ntk is not None:
            model = self.use_ntk_to_expend_input_token_length(model)

        if self.model_args.resize_emb is not None:
            # refer from https://zhuanlan.zhihu.com/p/656335338
            vocab_size_of_model = model.get_input_embeddings().weight.size(0)
            embedding_dim = model.get_input_embeddings().weight.size(1)
            vocab_size_of_tokenizer = len(self.tokenizer)
            self.logger.info(f'Vocab of the model: {vocab_size_of_model}')
            self.logger.info(f'Vocab of the tokenizer: {vocab_size_of_tokenizer}')
            if vocab_size_of_model != vocab_size_of_tokenizer:
                self.logger.info('Resize model embeddings to fit tokenizer')
                model.resize_token_embeddings(vocab_size_of_tokenizer)
                self.logger.info('Resize model lm_head to fit tokenizer')
                if self.model_args.model_type == 'chatglm' and any(key.endswith('rotary_pos_emb') for key, _ in model.named_modules()):
                    model.lm_head = model.transformer.output_layer
                lm_head = model.lm_head
                new_lm_head = torch.nn.Linear(in_features=embedding_dim, out_features=vocab_size_of_tokenizer, bias=False)
                new_lm_head.weight.data[:vocab_size_of_model, :] = lm_head.weight.data[:vocab_size_of_model, :]
                model.lm_head = new_lm_head.to(lm_head.weight.device).to(self.model_args.torch_dtype)

        if self.model_args.quantization_bit is not None and self.model_args.quantization == 'cpm':
            model = self.quantize(model, self.model_args.quantization_bit)

        if not self.is_deepspeed_train:
            model = dispatch(self.model_args.model_type, model, dispatched)

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

    def merge_lora_model(self):
        if self.model_args.checkpoint_dir is None:
            self.logger.error('checkpoint_dir is None.')
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
