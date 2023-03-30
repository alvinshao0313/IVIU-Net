from datetime import date
import sys

sys.path.append("XXXXXXXXXXXXXX")   # 此处输入项目根目录，本地运行可以注释掉 
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io as scio
from matplotlib import cm
from model.ParaSet import ParaSet
from model.Train import train
from model.GetResult import get_result
import argparse
import wandb

def run(data, trainFrom):
    if data == 'DC2':
        tabu = scio.loadmat('data/DC2/DC2_gt.mat')['X0_240']
        index = [18, 77, 131, 183, 208]
        size = 64
        if  trainFrom == 'train':
            train_from=False
            data_dir="data/DC2/DC2_64s_40dB_train_all_patch"
            epoch = 5000            # iterations
        else:
            train_from='logs/IVIU_DC2_min_loss.pt'
            epoch = 1
            data_dir="data/DC2/DC2_64s_40dB_test"
        para = ParaSet(
            data_dir=data_dir,
            lib_path='data/Library/USGS_Lib_240.mat',
            lib_symbol='Lib_240',
            length=64,
            width=64,
            admm_layers=40,
            train_from=train_from,
            batch_size=1,
            learning_rate=1e-4,
            epoch=epoch,
            save_dir='logs',
            network='IVIU',
            ckpt='logs/IVIU_DC2_min_loss.pt',
            date_name='DC2')
        
    elif data == 'DC3':
        tabu = scio.loadmat('data/DC3/DC3_128s_40dB.mat')['X0_240']
        index = [108, 8, 182, 15, 174]
        size = 128
        if  trainFrom == 'train':
            train_from=False
            epoch = 5000            # iterations
        else:
            train_from='logs/IVIU_DC3_min_loss.pt'
            epoch = 1
        para = ParaSet(
            data_dir="data/DC3/DC3_train_all_patch",
            lib_path='data/Library/USGS_Lib_240.mat',
            lib_symbol='Lib_240',
            length=128,
            width=128,
            admm_layers=40,
            train_from=train_from,
            batch_size=1,
            learning_rate=1e-4,
            epoch=epoch,
            save_dir='logs',
            network='IVIU',
            ckpt='logs/IVIU_DC3_final.pt',
            date_name='DC3')
    else:
        print('data 参数设置错误！')

    if not para.train_from and data == 'DC1':
        wandb.init(config={"noise": 40},
                project="DL_ADMM-DC1-para-test",
                entity="alvin-shao",
                name="DC1-{}ly-{}ep-{:.1e}lr".format(para.admm_layers,
                                                            para.epoch,
                                                            para.learning_rate),
                reinit=True)

    start_time = time.time()
    train(para)
    abuList, curveList, sreList = get_result(para)
    end_time = time.time()
    if para.train_from:
        abuList = np.array(abuList.cpu()).squeeze()
        sreList.append(np.mean(10 * np.log10(
        np.power(np.linalg.norm(tabu, ord=2, axis=0), 2) / np.power(np.linalg.norm(tabu - abuList,  ord=2, axis=0) + 1e-5, 2))))
    
    meanSre = sum(sreList)/len(sreList)
        
    print('Data: {}, run: {}, Time used: {:.6f}, SRE: {}, Mean SRE: {}'.format(data, trainFrom,
        end_time - start_time, sreList, meanSre))

    if para.train_from:
        plt.subplots(2, 5)
        j = 0
        for i in index:
            plt.subplot(2, 5, 1 + j)
            j += 1
            plt.imshow(tabu[i, :].reshape(size, size), cmap=cm.get_cmap('jet'))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('Original')
        j = 0
        for i in index:
            plt.subplot(2, 5, 6 + j)
            j += 1
            plt.imshow(abuList[i, :].reshape(size, size),
                    cmap=cm.get_cmap('jet'))
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('AL_ADMM')
        plt.savefig('./images/{}result.png'.format(date))

        save_name = './results/{}_IVIU_{:.6f}.mat'.format(data,
            sreList[0])
        mat_dic = {
            'est_abu': abuList,
            'rec_curve': curveList.cpu().detach().numpy().squeeze()
        }
        scio.savemat(save_name, mat_dic)
        

if __name__ == '__main__':
    parameters = argparse.ArgumentParser()
    parameters.add_argument('--tf', type=str, default='train')  # 'train' or 'test'
    parameters.add_argument('--data', type=str, default='DC2')  #'DC2' or 'DC3'
    args = parameters.parse_args()
    
    print(args)
    run(args.data, args.tf)
