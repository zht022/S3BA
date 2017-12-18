import numpy as np
from sto_adv_BA_algs import *


def Exp(alg, playtimes, c, other_alg_parameters, N, K, loss_generate, losses, mu, var, best_arm, turn_bud_to_N, verbose):
    '''
    This function play an algorithm for certain times and output the error rate.
    
    Inputs:
    - alg: Algorithm used to play;
    - playtimes;
    - other_alg_parameters: parameters used for AdS3BA;
    - verbose: If false, no output will be printed during training; else print information every verbose plays.

    Outputs:
    - error_rate.
    '''
    error = 0

    if (alg == Successive_Halving):
        print('Begin Successive Halving: {0} budgets-------------------------------' \
              .format(N))
        if loss_generate == False:
            for play in np.arange(1, playtimes+1):
                arm = Successive_Halving(N, K, loss_generate, losses[play-1, :, :], mu, var, turn_bud_to_N)
                if (arm != best_arm):
                    error += 1
                if (verbose and play % verbose == 0):
                    print('{0} plays finished ; up to now, {1} errors ' \
                          .format(play, error))
        else:
            for play in np.arange(1, playtimes+1):
                arm = Successive_Halving(N, K, loss_generate, [], mu, var, turn_bud_to_N)
                if (arm != best_arm):
                    error += 1
                if (verbose and play % verbose == 0):
                    print('{0} plays finished ; up to now, {1} errors ' \
                          .format(play, error))
    
    if (alg == Ad_UCBE):
        print('Begin Adaptive UCB-E: {0} budgets; exploration param c = {1} -------------------------------' \
              .format(N,c))
        if loss_generate == False:
            for play in np.arange(1, playtimes+1):
                arm = Ad_UCBE(c, N, K, loss_generate, losses[play-1, :, :], mu, var, turn_bud_to_N)
                if (arm != best_arm):
                    error += 1
                if (verbose and play % verbose == 0):
                    print('{0} plays finished ; up to now, {1} errors ' \
                          .format(play, error))
        else:
            for play in np.arange(1, playtimes+1):
                arm = Ad_UCBE(c, N, K, loss_generate, [], mu, var, turn_bud_to_N)
                if (arm != best_arm):
                    error += 1
                if (verbose and play % verbose == 0):
                    print('{0} plays finished ; up to now, {1} errors ' \
                          .format(play, error))
            
    if (alg == S3_BA):
        print('Begin S3-BA: {0} budgets; exploration param c = {1}; confidence param delta = {2} -------------------------------' \
              .format(N,other_alg_parameters[1],other_alg_parameters[0]))
        if loss_generate == False:
            for play in np.arange(1, playtimes+1):
                arm = S3_BA(other_alg_parameters, N, K, loss_generate, losses[play-1, :, :], mu, var, turn_bud_to_N)
                if (arm != best_arm):
                    error += 1
                if (verbose and play % verbose == 0):
                    print('{0} plays finished ; up to now, {1} errors ' \
                          .format(play, error))
        else:
            for play in np.arange(1, playtimes+1):
                arm = S3_BA(other_alg_parameters, N, K, loss_generate, [], mu, var, turn_bud_to_N)
                if (arm != best_arm):
                    error += 1
                if (verbose and play % verbose == 0):
                    print('{0} plays finished ; up to now, {1} errors ' \
                          .format(play, error))
        
    return float(error / playtimes)

