import numpy as np
import scipy.io as sio
from sto_adv_BA_algs import *
from sto_adv_BA_exp import *


playtime = 5000
K = 20

mu = 0.8 * np.ones([20, ])
var = np.zeros([20, ])
for i in range(20):
    mu[i] = mu[i] - i * 0.03
    var[i] = 0.3

error_rate_SH_T_ber = np.zeros([6, ])
for i in range(6):
    error_rate_SH_T_ber[i] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=500*(i+1),
                                 K=K, loss_generate=Bernoulli_loss, losses=[], mu=mu, var=[], turn_bud_to_N=True, verbose=False)
print(error_rate_SH_T_ber)
sio.savemat('./error_rate_SH_T_ber.mat', {'error_rate_SH_T_ber': error_rate_SH_T_ber})

error_rate_AdUCBE_T_ber = np.zeros([6, ])
for i in range(6):
    error_rate_AdUCBE_T_ber[i] = Exp(alg=Ad_UCBE, playtimes=playtime, c=0.5, other_alg_parameters=[], N=500*(i+1),
                                     K=K, loss_generate=Bernoulli_loss, losses=[], mu=mu, var=[], turn_bud_to_N=True, verbose=500)
print(error_rate_AdUCBE_T_ber)
sio.savemat('./error_rate_AdUCBE_T_ber.mat', {'error_rate_AdUCBE_T_ber': error_rate_AdUCBE_T_ber})

error_rate_AdUCBE_T_ber_2 = np.zeros([6, ])
for i in range(6):
    error_rate_AdUCBE_T_ber_2[i] = Exp(alg=Ad_UCBE, playtimes=playtime, c=2, other_alg_parameters=[], N=500*(i+1),
                                       K=K, loss_generate=Bernoulli_loss, losses=[], mu=mu, var=[], turn_bud_to_N=True, verbose=500)
print(error_rate_AdUCBE_T_ber_2)
sio.savemat('./error_rate_AdUCBE_T_ber_2.mat', {'error_rate_AdUCBE_T_ber_2': error_rate_AdUCBE_T_ber_2})

error_rate_S3BA_T_ber = np.zeros([6, ])
delta = 0.1
c = 1/2
C_w = 16
C_3 = 522
C_init = 1/9
C_gap = 60
parameters = [delta, c, C_w, C_3, C_init, C_gap]
for i in range(6):
    error_rate_S3BA_T_ber[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=500*(i+1),
                                   K=K, loss_generate=Bernoulli_loss, losses=[], mu=mu, var=[], turn_bud_to_N=True, verbose=500)
print(error_rate_S3BA_T_ber)
sio.savemat('./error_rate_S3BA_T_ber.mat', {'error_rate_S3BA_T_ber': error_rate_S3BA_T_ber})

error_rate_S3BA_T_ber_2 = np.zeros([6, ])
delta = 0.1
c = 2
C_w = 16
C_3 = 522
C_init = 1/9
C_gap = 60
parameters = [delta, c, C_w, C_3, C_init, C_gap]
for i in range(6):
    error_rate_S3BA_T_ber_2[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=500*(i+1),
                                     K=K, loss_generate=Bernoulli_loss, losses=[], mu=mu, var=[], turn_bud_to_N=True, verbose=500)
print(error_rate_S3BA_T_ber_2)
sio.savemat('./error_rate_S3BA_T_ber_2.mat', {'error_rate_S3BA_T_ber_2': error_rate_S3BA_T_ber_2})



error_rate_SH_T_norm = np.zeros([6, ])
for i in range(6):
    error_rate_SH_T_norm[i] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=500*(i+1),
                                  K=K, loss_generate=Gaussian_loss, losses=[], mu=mu, var=var, turn_bud_to_N=True, verbose=False)
print(error_rate_SH_T_norm)
sio.savemat('./error_rate_SH_T_norm.mat', {'error_rate_SH_T_norm': error_rate_SH_T_norm})

error_rate_AdUCBE_T_norm = np.zeros([6, ])
for i in range(6):
    error_rate_AdUCBE_T_norm[i] = Exp(alg=Ad_UCBE, playtimes=playtime, c=0.5, other_alg_parameters=[], N=500*(i+1),
                                      K=K, loss_generate=Gaussian_loss, losses=[], mu=mu, var=var, turn_bud_to_N=True, verbose=500)
print(error_rate_AdUCBE_T_norm)
sio.savemat('./error_rate_AdUCBE_T_norm.mat', {'error_rate_AdUCBE_T_norm': error_rate_AdUCBE_T_norm})

error_rate_AdUCBE_T_norm_2 = np.zeros([6, ])
for i in range(6):
    error_rate_AdUCBE_T_norm_2[i] = Exp(alg=Ad_UCBE, playtimes=playtime, c=2, other_alg_parameters=[], N=500*(i+1),
                                        K=K, loss_generate=Gaussian_loss, losses=[], mu=mu, var=var, turn_bud_to_N=True, verbose=500)
print(error_rate_AdUCBE_T_norm_2)
sio.savemat('./error_rate_AdUCBE_T_norm_2.mat', {'error_rate_AdUCBE_T_norm_2': error_rate_AdUCBE_T_norm_2})

error_rate_S3BA_T_norm = np.zeros([6, ])
delta = 0.1
c = 1/2
C_w = 16
C_3 = 522
C_init = 1/9
C_gap = 60
parameters = [delta, c, C_w, C_3, C_init, C_gap]
for i in range(6):
    error_rate_S3BA_T_norm[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=500*(i+1),
                                    K=K, loss_generate=Gaussian_loss, losses=[], mu=mu, var=var, turn_bud_to_N=True, verbose=200)
print(error_rate_S3BA_T_norm)
sio.savemat('./error_rate_S3BA_T_norm.mat', {'error_rate_S3BA_T_norm': error_rate_S3BA_T_norm})

error_rate_S3BA_T_norm_2 = np.zeros([6, ])
delta = 0.1
c = 2
C_w = 16
C_3 = 522
C_init = 1/9
C_gap = 60
parameters = [delta, c, C_w, C_3, C_init, C_gap]
for i in range(6):
    error_rate_S3BA_T_norm_2[i] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=500*(i+1),
                                      K=K, loss_generate=Gaussian_loss, losses=[], mu=mu, var=var, turn_bud_to_N=True, verbose=200)
print(error_rate_S3BA_T_norm_2)
sio.savemat('./error_rate_S3BA_T_norm_2.mat', {'error_rate_S3BA_T_norm_2': error_rate_S3BA_T_norm_2})

