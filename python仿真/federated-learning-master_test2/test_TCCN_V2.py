#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import scipy.io
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNMnist2
from models.Fed import FedAvg, outage_dec
from models.test import test_img
import scipy.io as scio

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.iid)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        trans_femnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_train = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=True, transform=trans_femnist)
        # dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=False, transform=trans_femnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset_train = datasets.CIFAR10('./dataset', train=True, transform=trans_cifar, download=True)  # 训练数据集
        # dataset_test = datasets.CIFAR10('./dataset', train=False, transform=trans_cifar, download=True)  # 测试数据集

        dataset_train = datasets.CIFAR10('./data/cifar-10-python', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar-10-python', train=False, download=False, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    Noiseless = 1
    Proposed = 1
    B_NoRIS = 1
    B_SR = 1
    B_u_RandRIS = 1
    men_num = 1

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        net_glob_opt = CNNCifar(args=args).to(args.device)
        net_glob_SR = CNNCifar(args=args).to(args.device)
        net_glob_NoRIS = CNNCifar(args=args).to(args.device)
        net_glob_u_RandRIS = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = CNNMnist2(num_classes=10,num_channels=1,batch_norm=True).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    net_glob_opt.train()
    net_glob_SR.train()
    net_glob_NoRIS.train()
    net_glob_u_RandRIS.train()

    # copy weights
    w_glob_opt = net_glob_opt.state_dict()
    w_glob_SR = net_glob_SR.state_dict()
    w_glob_NoRIS = net_glob_NoRIS.state_dict()
    w_glob_u_RandRIS = net_glob_u_RandRIS.state_dict()
    w_glob = net_glob.state_dict()


    active_users = ['Noiseless', 'Proposed', 'B_SR', 'B_NoRIS', 'B_u_RandRIS']

    # 生成每一个epoch的参与用户
    # 载入最优数据
    RIS_mun = 5 # RIS单元长度，要改
    data = scio.loadmat('./Result_RIS5x5.mat') # 读取文件
    args.sigma = 1# 噪声功率
    args.L = int(data['L'])# RIS单元个数
    args.N = int(data['M'])# BS天线个数
    ref = 1e+10 # 阔放因子
    args.M = int(data['K']) #用户个数
    W = data['W'] #功率分配向量
    G = data['G']
    Rican_UI = 3 + np.sqrt(12)  # device - RIS channel Rician factor
    Rican_UB = 3 + np.sqrt(12)  # direct channel Rician factor
    pathloss_UB = data['pathloss_UB']
    pathloss_UI = data['pathloss_UI']
    h_r_los = data['h_r_los']
    h_d_los = data['h_d_los']
    theta = data['theta']
    thetas = data['thetas']
    No_theta = np.zeros([args.L, 1], dtype=complex)
    rand_theta = np.exp(1j * np.random.randn(args.L, 1) * 2 * np.pi)  # 随机相位
    beta_pr = data['beta_pr']
    beta_pr = beta_pr.reshape(args.M)
    beta_ur = data['beta_ur']
    opt_beta = data['opt_beta']
    r_rand = data['r_rand']
    r_rand = r_rand.reshape(args.M)
    r_low = data['r_low']
    r_low = r_low.reshape(args.M)
    B_t = data['B_t']

    suc_opt = np.zeros(args.epochs * 1)
    suc_n_R = np.zeros(args.epochs * 1)
    suc_u_R = np.zeros(args.epochs * 1)
    suc_SR = np.zeros(args.epochs * 1 )
    # suc_p_R = np.zeros(args.epochs)
    # suc_u_sr_R = np.zeros(args.epochs)
    # suc_p_sr_R = np.zeros(args.epochs)
    # out_flag_opt = np.zeros([args.epochs, args.M])
    # out_flag_n_R = np.zeros([args.epochs, args.M])
    # out_flag_u_R = np.zeros([args.epochs, args.M])
    # out_flag_p_R = np.zeros([args.epochs, args.M])
    # out_flag_u_sr_R = np.zeros([args.epochs, args.M])
    # out_flag_p_sr_R = np.zeros([args.epochs, args.M])

    # B_p_RandRIS = 1
    # B_u_sr_RandRIS = 1
    # B_p_sr_RandRIS = 1

    # active_users = [5, 10, 15, 20, 25, 30, 35]
    # active_users = ['Noiseless', 'Proposed', 'B_NoRIS', 'B_u_RandRIS']

    out_flag = np.zeros([len(active_users), args.M, args.epochs * 1]) # 存放三维的中断flag，维度为beachmark个数*用户个数*args.epochs
    for iterA in range(args.epochs * 1):
        # define channel
        h_r = np.zeros([args.L, args.M], dtype=complex) # UE-RIS信道
        for m in range(args.M):
            h_r_nlos = np.sqrt(0.5) * (np.random.randn(args.L) + 1j * np.random.randn(args.L)) * ref ** 0.25 # 随机生成NLOS信道
            h_r[:, m] = pathloss_UI[:, m] * ((Rican_UI / (1 + Rican_UI)) ** 0.5 * h_r_los[:, m] + (
                    1 / (1 + Rican_UI)) ** 0.5 * h_r_nlos)

        h_d = np.zeros([args.N, args.M], dtype=complex) # UE-BS信道
        for m in range(args.M):
            h_d_nlos = np.sqrt(0.5) * (np.random.randn(args.N) + 1j * np.random.randn(args.N)) * ref ** 0.5 # 随机生成NLOS信道
            h_d[:, m] = pathloss_UB[:, m] * ((Rican_UB / (1 + Rican_UB)) ** 0.5 * h_d_los[:, m] + (
                    1 / (1 + Rican_UB)) ** 0.5 * h_d_nlos)

        rand_theta = np.exp(1j * np.random.randn(args.L, 1) * 2 * np.pi)  # 随机相位
        out_flag[0, :, iterA] = np.zeros(args.M) #完美CASE，中断都是0

        out_flag[1, :, iterA] = outage_dec(args, h_r, h_d, G, theta, opt_beta, B_t, r_low, W)
        # print(out_flag[1, :, iterA])
        # suc_opt = suc_opt + np.sum(out_flag_opt == 0)
        suc_opt[iterA] = np.sum(out_flag[1, :, iterA] == 0) #optCASE

        out_flag[2, :, iterA] = outage_dec(args, h_r, h_d, G, thetas, beta_ur, B_t, r_rand, W)
        suc_SR[iterA] = np.sum(out_flag[2, :, iterA] == 0)

        out_flag[3, :, iterA] = outage_dec(args, h_r, h_d, G, No_theta, beta_pr, B_t, r_rand, W)
        # suc_p_R = suc_p_R + np.sum(out_flag_p_R == 0)
        suc_n_R[iterA] = np.sum(out_flag[3, :, iterA] == 0)

        out_flag[4, :, iterA] = outage_dec(args, h_r, h_d, G, rand_theta, beta_ur, B_t, r_rand, W)
        # suc_u_R = suc_u_R + np.sum(out_flag_u_R == 0)
        suc_u_R[iterA] = np.sum(out_flag[4, :, iterA] == 0)

        # out_flag[4, :, iter] = outage_dec(args, h_r, h_d, G, rand_theta, beta_pr, B_t, r_rand, W)
        #
        # # suc_p_R = suc_p_R + np.sum(out_flag_p_R == 0)
        # suc_p_R[iter] = np.sum(out_flag[4, :, iter] == 0)
        # out_flag[5, :, iter] = outage_dec(args, h_r, h_d, G, thetas, beta_ur, B_t, r_rand, W)
        #
        # # suc_u_sr_R = suc_u_sr_R + np.sum(out_flag_u_sr_R == 0)
        # suc_u_sr_R[iter] = np.sum(out_flag[5, :, iter] == 0)
        # out_flag[6, :, iter] = outage_dec(args, h_r, h_d, G, thetas, beta_pr, B_t, r_rand, W)
        #
        # # suc_p_sr_R = suc_p_sr_R + np.sum(out_flag_p_sr_R == 0)
        # suc_p_sr_R[iter] = np.sum(out_flag[6, :, iter] == 0)
    # print('out_flag[:, :, 0]', out_flag[:, :, 0])
    # print('out_flag_opt', out_flag_opt)
    # print('out_flag_n_R', out_flag_n_R)
    # print('out_flag_u_R', out_flag_u_R)
    # print('out_flag_p_R', out_flag_p_R)
    # print('out_flag_u_sr_R', out_flag_u_sr_R)
    # print('out_flag_p_sr_R', out_flag_p_sr_R)

    print('suc_opt', suc_opt)
    print('suc_SR', suc_SR)
    print('suc_n_R', suc_n_R)
    print('uc_u_R', suc_u_R)
    # print('suc_p_R', suc_p_R)
    # print('suc_u_sr_R', suc_u_sr_R)
    # print('suc_p_sr_R', suc_p_sr_R)

    print('mean suc_opt', np.mean(suc_opt))
    print('mean suc_SR', np.mean(suc_SR))
    print('mean suc_n_R', np.mean(suc_n_R))
    print('mean suc_u_R', np.mean(suc_u_R))
    # print('mean suc_p_R', np.mean(suc_p_R))
    # print('mean suc_u_sr_R', np.mean(suc_u_sr_R))
    # print('mean suc_p_sr_R', np.mean(suc_p_sr_R))
    result_file_one = './save/Device_every_iter{}x{}.mat'.format(RIS_mun, RIS_mun)
    scipy.io.savemat(result_file_one, mdict={'out_flag': out_flag, 'suc_opt': suc_opt, 'suc_SR': suc_SR, 'suc_n_R': suc_n_R, 'suc_u_R': suc_u_R})

    # training
    #
    # cv_loss, cv_acc = [], []
    # val_loss_pre, counter = 0, 0
    # net_best = None
    # best_loss = None
    # val_acc_list, net_list = [], []

    # print('len(active_users) ', len(active_users))
    loss_matrix = np.zeros([len(active_users), args.epochs])
    acc_matrix = np.zeros([len(active_users), args.epochs])

    if Noiseless:
        loss_test = []
        accuracy_test = []
        lr = copy.deepcopy(args.lr)

        for iterB in range(args.epochs):
            # print('lr', lr)
            loss_locals = []
            w_locals = []
            out_arr = out_flag[0, :, iterB]
            idxs_users = []
            for kk in range(args.M):
                if out_arr[kk] == 0:
                    idxs_users.append(kk)
            # print('out_arr', out_arr)
            # print('idxs_users', idxs_users)
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = range(active_users[active_index])
            # print('iter1{}的lr为{}'.format(iter1, lr))
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=copy.deepcopy(dict_users[idx]))
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=lr)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            if len(w_locals) != 0:
                w_glob = FedAvg(w_locals)
            else:
                w_glob = net_glob.state_dict()

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)

            # testing
            net_glob.eval()
            acc_train, loss_train_test = test_img(net_glob, dataset_train, args)
            print('Round {:3d}, Average loss {:.3f}'.format(iterB, loss_train_test))
            loss_test.append(loss_train_test)
            acc_test, loss_test2 = test_img(net_glob, dataset_test, args)
            accuracy_test.append(acc_test.tolist())
            # print(accuracy_test)
            # print("Training loss: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            # lr = args.lr / (iterB + 1)
        # print('loss_test', loss_test)
        # print('accuracy_test', accuracy_test)
        loss_matrix[0, :] = np.array(loss_test)
        acc_matrix[0, :] = np.array(accuracy_test)
    if Proposed:
        loss_test_opt = []
        accuracy_test_opt = []
        lr_opt = copy.deepcopy(args.lr)
        for iterB in range(args.epochs):
            # print('lr_opt', lr_opt)
            loss_locals_opt = []
            w_locals_opt = []
            out_arr_opt = out_flag[1, :, iterB]
            idxs_users_opt = []
            for kk in range(args.M):
                if out_arr_opt[kk] == 0:
                    idxs_users_opt.append(kk)
            # print('out_arr', out_arr)
            # print('idxs_users_opt', idxs_users_opt)
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = range(active_users[active_index])
            # print('iter1{}的lr为{}'.format(iter1, lr))
            for idx in idxs_users_opt:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=copy.deepcopy(dict_users[idx]))
                w, loss = local.train(net=copy.deepcopy(net_glob_opt).to(args.device), lr=lr_opt)
                w_locals_opt.append(copy.deepcopy(w))
                loss_locals_opt.append(copy.deepcopy(loss))
            # update global weights
            if len(w_locals_opt) != 0:
                w_glob_opt = FedAvg(w_locals_opt)
            else:
                w_glob_opt = net_glob_opt.state_dict()

            # copy weight to net_glob
            net_glob_opt.load_state_dict(w_glob_opt)

            # print loss
            loss_avg_opt = sum(loss_locals_opt) / len(loss_locals_opt)

            # testing
            net_glob_opt.eval()
            acc_train_opt, loss_train_test_opt = test_img(net_glob_opt, dataset_train, args)
            print('Round {:3d}, Average loss {:.3f} WITH OPT'.format(iterB, loss_train_test_opt))
            loss_test_opt.append(loss_train_test_opt)
            acc_test_opt, loss_test2_opt = test_img(net_glob_opt, dataset_test, args)
            accuracy_test_opt.append(acc_test_opt.tolist())
            # print(accuracy_test)
            # print("Training loss: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f} WITH OPT".format(acc_test_opt))
            # lr_opt = args.lr / (iterB + 1)
        # print('loss_test_opt', loss_test_opt)
        # print('accuracy_test_opt', accuracy_test_opt)
        loss_matrix[1, :] = np.array(loss_test_opt)
        acc_matrix[1, :] = np.array(accuracy_test_opt)
    if B_SR:
        loss_test_SR = []
        accuracy_test_SR = []
        lr_SR = copy.deepcopy(args.lr)

        for iterB in range(args.epochs):
            # print('lr_SR', lr_SR)
            loss_locals_SR = []
            w_locals_SR = []
            out_arr_SR = out_flag[2, :, iterB]
            idxs_users_SR = []
            for kk in range(args.M):
                if out_arr_SR[kk] == 0:
                    idxs_users_SR.append(kk)
            # print('out_arr', out_arr)
            # print('idxs_users_SR', idxs_users_SR)
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = range(active_users[active_index])
            # print('iter1{}的lr为{}'.format(iter1, lr))
            for idx in idxs_users_SR:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=copy.deepcopy(dict_users[idx]))
                w, loss = local.train(net=copy.deepcopy(net_glob_SR).to(args.device), lr=lr_SR)
                w_locals_SR.append(copy.deepcopy(w))
                loss_locals_SR.append(copy.deepcopy(loss))
            # update global weights
            if len(w_locals_SR) != 0:
                w_glob_SR = FedAvg(w_locals_SR)
            else:
                w_glob_SR = net_glob_SR.state_dict()

            # copy weight to net_glob
            net_glob_SR.load_state_dict(w_glob_SR)

            # print loss
            loss_avg_SR = sum(loss_locals_SR) / len(loss_locals_SR)

            # testing
            net_glob_SR.eval()
            acc_train_SR, loss_train_test_SR = test_img(net_glob_SR, dataset_train, args)
            print('Round {:3d}, Average loss {:.3f} with max SR'.format(iterB, loss_train_test_SR))
            loss_test_SR.append(loss_train_test_SR)
            acc_test_SR, loss_test2_SR = test_img(net_glob_SR, dataset_test, args)
            accuracy_test_SR.append(acc_test_SR.tolist())
            # print(accuracy_test)
            # print("Training loss: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f} with max SR".format(acc_test_SR))
            # lr_SR = args.lr / (iterB + 1)
        # print('loss_test_SR', loss_test_SR)
        # print('accuracy_test_SR', accuracy_test_SR)
        loss_matrix[2, :] = np.array(loss_test_SR)
        acc_matrix[2, :] = np.array(accuracy_test_SR)
    if B_NoRIS:
        loss_test_NoRIS = []
        accuracy_test_NoRIS = []
        lr_NoRIS = copy.deepcopy(args.lr)

        for iterB in range(args.epochs):
            # print('lr_NoRIS', lr_NoRIS)
            loss_locals_NoRIS = []
            w_locals_NoRIS = []
            out_arr_NoRIS = out_flag[3, :, iterB]
            idxs_users_NoRIS = []
            for kk in range(args.M):
                if out_arr_NoRIS[kk] == 0:
                    idxs_users_NoRIS.append(kk)
            # print('out_arr', out_arr)
            # print('idxs_users_NoRIS', idxs_users_NoRIS)
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = range(active_users[active_index])
            # print('iter1{}的lr为{}'.format(iter1, lr))
            for idx in idxs_users_NoRIS:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=copy.deepcopy(dict_users[idx]))
                w, loss = local.train(net=copy.deepcopy(net_glob_NoRIS).to(args.device), lr=lr_NoRIS)
                w_locals_NoRIS.append(copy.deepcopy(w))
                loss_locals_NoRIS.append(copy.deepcopy(loss))
            # update global weights
            if len(w_locals_NoRIS) != 0:
                w_glob_NoRIS = FedAvg(w_locals_NoRIS)
            else:
                w_glob_NoRIS = net_glob_NoRIS.state_dict()

            # copy weight to net_glob
            net_glob_NoRIS.load_state_dict(w_glob_NoRIS)

            # print loss
            loss_avg_NoRIS = sum(loss_locals_NoRIS) / len(loss_locals_NoRIS)

            # testing
            net_glob_NoRIS.eval()
            acc_train_NoRIS, loss_train_test_NoRIS = test_img(net_glob_NoRIS, dataset_train, args)
            print('Round {:3d}, Average loss {:.3f} with NoRIS'.format(iterB, loss_train_test_NoRIS))
            loss_test_NoRIS.append(loss_train_test_NoRIS)
            acc_test_NoRIS, loss_test2_NoRIS = test_img(net_glob_NoRIS, dataset_test, args)
            accuracy_test_NoRIS.append(acc_test_NoRIS.tolist())
            # print(accuracy_test)
            # print("Training loss: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f} with NoRIS".format(acc_test_NoRIS))
            # lr_NoRIS = args.lr / (iterB + 1)
        # print('loss_test_NoRIS', loss_test_NoRIS)
        # print('accuracy_test_NoRIS', accuracy_test_NoRIS)
        loss_matrix[3, :] = np.array(loss_test_NoRIS)
        acc_matrix[3, :] = np.array(accuracy_test_NoRIS)
    if B_u_RandRIS:
        loss_test_u_RandRIS = []
        accuracy_test_u_RandRIS = []
        lr_u_RandRIS = copy.deepcopy(args.lr)

        for iterB in range(args.epochs):
            print('lr_u_RandRIS', lr_u_RandRIS)
            loss_locals_u_RandRIS = []
            w_locals_u_RandRIS = []
            out_arr_u_RandRIS = out_flag[4, :, iterB]
            idxs_users_u_RandRIS = []
            for kk in range(args.M):
                if out_arr_u_RandRIS[kk] == 0:
                    idxs_users_u_RandRIS.append(kk)
            # print('out_arr', out_arr)
            # print('idxs_users_u_RandRIS', idxs_users_u_RandRIS)
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = range(active_users[active_index])
            # print('iter1{}的lr为{}'.format(iterB, lr))
            for idx in idxs_users_u_RandRIS:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=copy.deepcopy(dict_users[idx]))
                w, loss = local.train(net=copy.deepcopy(net_glob_u_RandRIS).to(args.device), lr=lr_u_RandRIS)
                w_locals_u_RandRIS.append(copy.deepcopy(w))
                loss_locals_u_RandRIS.append(copy.deepcopy(loss))
            # update global weights
            if len(w_locals_u_RandRIS) != 0:
                w_glob_u_RandRIS = FedAvg(w_locals_u_RandRIS)
            else:
                w_glob_u_RandRIS = net_glob_u_RandRIS.state_dict()

            # copy weight to net_glob
            net_glob_u_RandRIS.load_state_dict(w_glob_u_RandRIS)

            # print loss
            loss_avg_u_RandRIS = sum(loss_locals_u_RandRIS) / len(loss_locals_u_RandRIS)

            # testing
            net_glob_u_RandRIS.eval()
            acc_train_u_RandRIS, loss_train_test_u_RandRIS = test_img(net_glob_u_RandRIS, dataset_train, args)
            print('Round {:3d}, Average loss {:.3f} with u_RandRIS'.format(iterB, loss_train_test_u_RandRIS))
            loss_test_u_RandRIS.append(loss_train_test_u_RandRIS)
            acc_test_u_RandRIS, loss_test2_u_RandRIS = test_img(net_glob_u_RandRIS, dataset_test, args)
            accuracy_test_u_RandRIS.append(acc_test_u_RandRIS.tolist())
            # print(accuracy_test)
            # print("Training loss: {:.2f}".format(acc_train))
            print("Testing accuracy with u_RandRIS: {:.2f}".format(acc_test_u_RandRIS))
            # lr_u_RandRIS = args.lr / (iterB + 1)
        # print('loss_test_u_RandRIS', loss_test_u_RandRIS)
        # print('accuracy_test_u_RandRIS', accuracy_test_u_RandRIS)
        loss_matrix[4, :] = np.array(loss_test_u_RandRIS)
        acc_matrix[4, :] = np.array(accuracy_test_u_RandRIS)

    result_file_two = './save/Performance{}x{}.mat'.format(RIS_mun, RIS_mun)
    scipy.io.savemat(result_file_two, mdict={'loss_matrix': loss_matrix, 'acc_matrix': acc_matrix})
    # print(loss_set)
    # print(acc_set)
    # print(loss_matrix)
    # print(acc_matrix)
    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)



