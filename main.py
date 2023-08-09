# -*- coding: utf-8 -*-
# @Time : 2023/7/11 23:39
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
from engines.utils.check_load_config import Configure
from engines.data import DataManager
from loguru import logger
from config import mode
from engines.train import Train
from engines.predict import Predictor


if __name__ == '__main__':

    config = Configure()
    log_name = config.training_args.output_dir + '/logs/' + mode + '.log'
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
        pass
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
        predict = Predictor(data_manager, config, logger)
        predict.merge_peft_model()
    elif mode == 'show_model_info':
        # 打印模型参数
        predictor = Predictor(data_manager, config, logger)
        predictor.show_model_info()
    elif mode == 'save_quantized_model':
        # 存储量化的模型
        if config.model_args.quantization_bit not in (4, 8):
            raise ValueError('Quantization bit not set.')
        predictor = Predictor(data_manager, config, logger)
        predictor.save_quantized_model()
    elif mode == 'sft_batch_test':
        # 模型效果测试
        predictor = Predictor(data_manager, config, logger)
        predictor.sft_batch_test()
    elif mode == 'rm_batch_test':
        # 奖励模型效果测试
        predictor = Predictor(data_manager, config, logger)
        predictor.rm_batch_test()
