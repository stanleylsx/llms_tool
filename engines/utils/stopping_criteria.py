# -*- coding: utf-8 -*-
# @Time : 2023/9/19 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : stopping_criteria.py
# @Software: PyCharm
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings


class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):
        self.token_id_list = token_id_list

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids, scores, **kwargs):
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


def add_stopping_criteria(token_id_list):
    """
    You can define your own stop token here.
    """
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list))
