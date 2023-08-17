# -*- coding: utf-8 -*-
# @Time : 2023/7/19 21:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm
from rouge_chinese import Rouge
import numpy as np
import jieba


class Metrics:
    def __init__(self, data_manager, logger):
        self.data_manager = data_manager
        self.tokenizer = data_manager.tokenizer
        self.rouge = Rouge()
        self.logger = logger

    def computer_supervised_fine_tuning_metric(self, eval_preds):
        preds, labels = eval_preds
        score_dict = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
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
    def computer_training_reward_metric(eval_preds):
        preds, _ = eval_preds
        accuracy = np.array(preds[0] > preds[1]).sum() / len(preds[0])
        return {'accuracy': accuracy}
