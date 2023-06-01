#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
import math

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def outage_dec(args, h_r, h_d, G, RIS_theta, beta, Band, rate, W):
    # opt out flag
    h_opt = np.zeros([args.N, args.M], dtype=complex)
    g_ca = np.zeros([args.N, args.L, args.M], dtype=complex)
    for i in range(args.M):
        g_ca[:, :, i] = G @ np.diag(h_r[:, i])
        h_opt[:, i] = ((h_d[:, i]).reshape(args.N, 1) + g_ca[:, :, i] @ RIS_theta).reshape(args.N)
    capability = np.zeros([args.M])
    for i in range(args.M):
        capability[i] = beta[i] * Band * math.log2(1 + (W[i] * (np.linalg.norm(h_opt[:, i])) ** 2) / args.sigma)
    # print('Transmitting rate with opt IRS', rate)
    # print('Channel capability with opt IRS', capability)
    out_flag = np.zeros(args.M)
    for i in range(args.M):
        # print('Channel capability with opt IRS', capability[i])
        # print('Transmitting rate with opt IRS', rate[i])
        if capability[i] >= rate[i]:  # Blog(1+SNR)>=R,不中断，否则终中断
            out_flag[i] = 0  # =1，中断；=0，不中断
        else:
            out_flag[i] = 1
    return out_flag