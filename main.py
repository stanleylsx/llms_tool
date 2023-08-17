# -*- coding: utf-8 -*-
# @Time : 2023/7/11 23:39
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
from engines.utils.check_load_config import Configure
from engines.data import DataManager
from loguru import logger
from engines.models import BaseModels
from engines.train import Train
from engines.predict import Predictor


if __name__ == '__main__':

    config = Configure()
    mode = config.mode
    log_name = './logs/' + mode + '.log'
    logger.add(log_name, encoding='utf-8')

    data_manager = DataManager(config, logger)
    if mode == 'pretrain':
        # 模型预训练
        pass
    elif mode == 'sft_train':
        # 模型指令微调
        train = Train(data_manager, config, logger)
        train.supervised_fine_tuning()
    elif mode == 'rm_train':
        # 奖励模型训练
        train = Train(data_manager, config, logger)
        train.train_reward_model()
    elif mode == 'ppo_train':
        # 奖励模型强化训练
        train = Train(data_manager, config, logger)
        train.train_ppo()
    elif mode == 'sft_batch_test':
        # 微调模型效果测试
        train = Train(data_manager, config, logger)
        train.supervised_fine_tuning(test=True)
    elif mode == 'rm_batch_test':
        # 奖励模型效果测试
        train = Train(data_manager, config, logger)
        train.train_reward_model(test=True)
    elif mode == 'web_inference':
        # 网页端测试模型
        predict = Predictor(data_manager, config, logger)
        predict.web_inference()
    elif mode == 'terminal_inference':
        # 终端模型交互
        predict = Predictor(data_manager, config, logger)
        predict.terminal_inference()
    elif mode == 'merge_peft_model':
        # 融合模型
        model = BaseModels(data_manager, config, logger)
        model.merge_peft_model()
    elif mode == 'show_model_info':
        # 打印模型参数
        model = BaseModels(data_manager, config, logger)
        model.show_model_info()
    elif mode == 'save_quantized_model':
        # 存储量化的模型
        if config.model_args.quantization_bit not in (4, 8):
            raise ValueError('Quantization bit not set.')
        model = BaseModels(data_manager, config, logger)
        model.save_quantized_model()
