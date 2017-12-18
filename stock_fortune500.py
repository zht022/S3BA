import numpy as np
from sto_adv_BA_algs import *
import pandas as pd

f_500_stock_csv = pd.read_csv("./fortune-500.csv", encoding = 'gbk')
f_500_stock = f_500_stock_csv.iloc[:, :].values.T

amountToInvest = 1
stock_payoff = np.zeros([9, 2607])
stock_cum_reward = np.zeros([9, 2607])
stock_mean = np.zeros([9, 2607])
for i in range(1, 10):
    stock_payoff[i - 1, 0] = f_500_stock[(2 * i), 0] * amountToInvest / f_500_stock[(2 * i - 1), 0] - amountToInvest
    stock_cum_reward[i - 1, 0] = f_500_stock[(2 * i), 0] * amountToInvest / f_500_stock[(2 * i - 1), 0] - amountToInvest
    for j in range(1, 2607):
        stock_payoff[i - 1, j] = f_500_stock[(2 * i), j] * amountToInvest / f_500_stock[(2 * i - 1), j] - amountToInvest
        stock_cum_reward[i - 1, j] = stock_cum_reward[i - 1, j - 1] + stock_payoff[i - 1, j]
        stock_mean[i - 1, j] = stock_cum_reward[i - 1, j] / (j + 1)
stock_loss = - stock_mean


print('Using Successive Halving ... ... ')
i = 0
n = np.arange(500, 2401, 100)
choice_SH = np.zeros([len(n), ])
real_SH = np.zeros([len(n), ])
for N in n:
    N = int(N)
    choice_SH[i] = Successive_Halving(N, 9, False, stock_loss, [], [], True)
    real_SH[i] = np.sum(stock_mean[:, (N):(N + 200)], 1).argmax()
    i += 1
print(choice_SH)
print(real_SH + 1)


print('Using AdUCBE ... ... ')
i = 0
n = np.arange(500, 2401, 100)
choice_AdUCBE = np.zeros(len(n), )
real_AdUCBE = np.zeros(len(n), )
for N in n:
    choice_AdUCBE[i] = Ad_UCBE(2, N, 9, False, stock_loss, [], [], True)
    real_AdUCBE[i] = np.sum(stock_mean[:, (N):(N + 200)], 1).argmax()
    i += 1
print(choice_AdUCBE)
print(real_AdUCBE + 1)


print('Using AdS3BA ... ... ')
i = 0
n = np.arange(500, 2401, 100)
choice_AdS3BA = np.zeros(len(n), )
real_AdS3BA = np.zeros(len(n), )
delta, c, C_w, C_3, C_init, C_gap = 0.1, 2, 2e-6, 2e-3, 1, 60
parameters = [delta, c, C_w, C_3, C_init, C_gap]
for N in n:
    choice_AdS3BA[i] = S3_BA(parameters, N, 9, False, stock_loss, [], [], True)
    real_AdS3BA[i] = np.sum(stock_mean[:, (N):(N + 200)], 1).argmax()
    i += 1
print(choice_AdS3BA)
print(real_AdS3BA + 1)