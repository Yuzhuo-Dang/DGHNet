# coding: utf-8
# @email: dangyuzhuo@nudt.edu.cn

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product

import numpy as np

from src.utils.dataset import RecDataset
from src.utils.dataloader import TrainDataLoader, EvalDataLoader
from src.utils.logger import init_logger
from src.utils.configurator import Config
from src.utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict) #模型和数据集参数
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config) #输入日志

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset)) #数据的item、user、交互数量

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True) #2048
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']), #4096
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    hyper_ret_test = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters 超参数
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators: #超参数的组合（网格搜索）
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup() #数据预处理 随机化数据
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device']) #模型加载和初始化
        logger.info(model)

        #学习学习数据集进行划分
        #model.pre_processing(config, logger)
        #logger.info('con data ready')

        # trainer loading and initialization
        trainer = get_trainer()(config, model) #common/trainer.py
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid, best_valid_test_result, best_test_result, item = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid)) #保存此超参数下的结果
        hyper_ret_test.append((hyper_tuple, best_valid_test_result, best_test_result))  # 保存此超参数下的结果

        # save best test
        if best_test_result[val_metric] > best_test_value:
            best_test_value = best_test_result[val_metric] #保留了测试集上R@20最优的数据
            best_test_idx = idx
            np.save("C:/MS/paper/3/MM/src/log/visual/D_babyD.npy", item)
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret_test:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}\n\n\n'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret_test[best_test_idx][0],
                                                                   dict2str(hyper_ret_test[best_test_idx][1]),
                                                                   dict2str(hyper_ret_test[best_test_idx][2])))

