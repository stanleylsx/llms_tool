# -*- coding: utf-8 -*-
# @Time : 2023/7/19 21:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm
from rouge_chinese import Rouge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import jieba


class Metrics:
    def __init__(self, data_manager, logger):
        self.data_manager = data_manager
        self.tokenizer = data_manager.tokenizer
        self.rouge = Rouge()
        self.logger = logger

    def computer_supervised_fine_tuning_metric(self, preds, labels):
        score_dict = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        for pred, label in zip(preds, labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            if len(' '.join(hypothesis).split()) == 0 or len(' '.join(reference).split()) == 0:
                result = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
            else:
                scores = self.rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
                result = scores[0]
            for k, v in result.items():
                score_dict[k].append(round(v['f'] * 100, 4))
        metric_results = {}
        for k, v in score_dict.items():
            metric_results[k] = float(np.mean(v))
        return metric_results

    @staticmethod
    def computer_training_reward_metric(preds, labels):
        # MSE
        mse = mean_squared_error(labels, preds)
        # MAE
        mae = mean_absolute_error(labels, preds)
        # accuracy
        accuracy = (preds[0] > preds[1]).sum() / len(preds[0])

        return {'mse': mse, 'mae': mae, 'accuracy': accuracy}
