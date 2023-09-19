# -*- coding: utf-8 -*-
# @Time : 2023/9/19 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : logits_process.py
# @Software: PyCharm
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList


def logits_processor():
    # https://huggingface.co/docs/transformers/v4.33.2/en/internal/generation_utils#transformers.LogitsProcessor
    # You can define your logits processor here to control the generate process.
    # https://blog.csdn.net/weixin_44826203/article/details/129928897
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor
