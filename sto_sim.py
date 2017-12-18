import numpy as np
import scipy.io as sio
from sto_adv_BA_algs import *
from sto_adv_BA_exp import *

playtime = 5000
K = 20

def set_mu():
    mu = {}

    u = 0.4 * np.ones([20, ])
    u[0] = 0.5
    mu[1] = u

    u = 0.5 * np.ones([20, ])
    for i in range(1, 10):
        u[i] = 0.42
    for i in range(10, 20):
        u[i] = 0.38
    mu[2] = u

    u = 0.5 * np.ones([20, ])
    u[1] = 0.48
    for i in range(2, 20):
        u[i] = 0.38
    mu[3] = u

    u = 0.5 * np.ones([20, ])
    for i in range(1, 10):
        u[i] = 0.5 - 1 / (5 * K)
    for i in range(10, 20):
        u[i] = 0.25
    mu[4] = u

    return mu

mu = set_mu() # mu is a dictionary contains different mu's for each simulation

Budgets = 3000
er_SH_sto = np.zeros([4, ])
er_AdUCBE_sto = np.zeros([4, ]) # c = 0.25
er_AdUCBE2_sto = np.zeros([4, ]) # c = 2
er_S3BA_sto = np.zeros([4, ]) # delta = 0.1
er_S3BA2_sto = np.zeros([4, ])
er_S3BA_sdelta_sto = np.zeros([4, ]) # delta = 0.05
er_S3BA_sdelta2_sto = np.zeros([4, ])

def set_params():
    parameters = {}

    delta = 0.1
    c = 0.25
    C_w = 16
    C_3 = 522
    C_init = 1/9
    C_gap = 60
    parameters[1] = [delta, c, C_w, C_3, C_init, C_gap]

    delta = 0.1
    c = 2
    parameters[2] = [delta, c, C_w, C_3, C_init, C_gap]

    delta = 0.05
    c = 0.25
    parameters[3] = [delta, c, C_w, C_3, C_init, C_gap]

    delta = 0.05
    c = 2
    parameters[4] = [delta, c, C_w, C_3, C_init, C_gap]

    return parameters


parameters = set_params() # set parameters for S3BA alg.


print('Algorithm ' + str(1))
for i in range(4):
    print('Group ' + str(i+1))
    er_SH_sto[i] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=Budgets, K=K,
                       loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=1000)
sio.savemat('./er_SH_sto.mat', {'er_SH_sto': er_SH_sto})

print('Algorithm ' + str(2))
for i in range(4):
    print('Group ' + str(i + 1))
    er_AdUCBE_sto[i] = Exp(alg=Ad_UCBE, playtimes=playtime, c=0.25, other_alg_parameters=[], N=Budgets, K=K,
                           loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=500)
sio.savemat('./er_AdUCBE_sto.mat', {'er_AdUCBE_sto': er_AdUCBE_sto})

print('Algorithm ' + str(3))
for i in range(4):
    print('Group ' + str(i + 1))
    er_AdUCBE2_sto[i] = Exp(alg=Ad_UCBE, playtimes=playtime, c=2, other_alg_parameters=[], N=Budgets, K=K,
                           loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=500)
sio.savemat('./er_AdUCBE2_sto.mat', {'er_AdUCBE2_sto': er_AdUCBE2_sto})

print('Algorithm ' + str(4))
for i in range(4):
    print('Group ' + str(i + 1))
    er_S3BA_sto[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters[1], N=Budgets, K=K,
                         loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=500)
sio.savemat('./er_S3BA_sto.mat', {'er_S3BA_sto': er_S3BA_sto})

print('Algorithm ' + str(5))
for i in range(4):
    print('Group ' + str(i + 1))
    er_S3BA2_sto[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters[2], N=Budgets, K=K,
                          loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=500)
sio.savemat('./er_S3BA2_sto.mat', {'er_S3BA2_sto': er_S3BA2_sto})

print('Algorithm ' + str(6))
for i in range(4):
    print('Group ' + str(i + 1))
    er_S3BA_sdelta_sto[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters[3], N=Budgets, K=K,
                                loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=500)
sio.savemat('./er_S3BA_sdelta_sto.mat', {'er_S3BA_sdelta_sto': er_S3BA_sdelta_sto})

print('Algorithm ' + str(7))
for i in range(4):
    print('Group ' + str(i + 1))
    er_S3BA_sdelta2_sto[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters[4], N=Budgets, K=K,
                                 loss_generate=Bernoulli_loss, losses=[], mu=mu[i + 1], var=[], turn_bud_to_N=True, verbose=500)
sio.savemat('./er_S3BA_sdelta2_sto.mat', {'er_S3BA_sdelta2_sto': er_S3BA_sdelta2_sto})








