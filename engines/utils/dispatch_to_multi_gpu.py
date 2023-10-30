# -*- coding: utf-8 -*-
# @Time : 2023/9/21 21:30
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : dispatch_to_multi_gpu.py
# @Software: PyCharm
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
import torch


def dispatch(model_type, model, dispatched):
    if dispatched:
        return model
    if model_type == 'chatglm':
        model.tie_weights()
        device_map = infer_chatglm_device_map(model)
    else:
        kwargs = {'dtype': model.dtype, 'no_split_module_classes': model._no_split_modules}
        max_memory = get_balanced_memory(model, **kwargs)
        model.tie_weights()
        device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)
    model = dispatch_model(model, device_map=device_map)
    return model


def infer_chatglm_device_map(model):
    num_gpus = torch.cuda.device_count()
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2和ChatGLM3
    chatglm2and3 = False
    for key, _ in model.named_modules():
        if key.endswith('rotary_pos_emb'):
            chatglm2and3 = True
            break

    if chatglm2and3:
        device_map = {
            'transformer.embedding.word_embeddings': 0,
            'transformer.encoder.final_layernorm': 0,
            'transformer.output_layer': 0,
            'transformer.rotary_pos_emb': 0,
            'lm_head': 0
        }
    else:
        device_map = {
            'transformer.word_embeddings': 0,
            'transformer.final_layernorm': 0,
            'lm_head': 0
        }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        if chatglm2and3:
            device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        else:
            device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map
