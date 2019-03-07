import numpy as np

def Bernoulli_loss(mu, r):
    '''
    Generate Bernoulli loss sequence.

    Inputs:
    - mu: Mean;
    - r: Length of the generated sequence.

    Outputs:
    - Random loss sequence.
    '''
    return np.random.rand(int(r)) > mu


def Gaussian_loss(mu, var, r):
    '''
    Generate Gaussian loss sequence.

    Inputs:
    - mu: Mean;
    - var: Variance;
    - r: Length of the generated sequence.

    Outputs:
    - rand_loss: Random loss sequence.
    '''
    return np.ones([int(r), ]) - mu + np.random.randn(int(r)) * var


def get_reward_AdUCBE(loss_generate, loss, mu, var, X, It, t):
    reward = 0
    if loss_generate == False:
        reward = 1 - loss[It, t]
        X[It] += reward
    if loss_generate == Bernoulli_loss:
        reward = 1 - loss_generate(mu[It], 1)
        X[It] += reward
    if loss_generate == Gaussian_loss:
        reward = 1 - loss_generate(mu[It], var[It], 1)
        X[It] += reward
    return X, reward


def predef_AdUCBE(N, K, turn_bud_to_N):
    ## Some definitions of symbols in the algorithm
    barLog = 0.5
    for i in range(2, K + 1):
        barLog += 1. / i

    n = [np.ceil((N - K) / (barLog * (K + 1 - k))) for k in range(1, K)]
    n = np.asarray(n).astype(np.int32)

    t = np.zeros([K, ])
    t[1] = K * n[0]
    tmp = 0
    for k in range(2, K):
        tmp += n[k - 2]
        t[k] = tmp + (K - k + 1) * n[k - 1]
    t = t.astype(np.int32)

    if turn_bud_to_N:
        bud = N
        N = np.ceil(1.0 * N ** 2 / t[-1])
        while (1):
            n = [np.ceil((N - K) / (barLog * (K + 1 - k))) for k in range(1, K)]
            n = np.asarray(n).astype(np.int32)
            t[1] = K * n[0]
            tmp = 0
            for k in range(2, K):
                tmp += n[k - 2]
                t[k] = tmp + (K - k + 1) * n[k - 1]
            t = t.astype(np.int32)
            if t[-1] > bud:
                N -= 1
            else:
                break
    return barLog, n, t, N


def get_loss_SH(loss_generate, loss, mu, var, S_k, size_Sk, R, r):
    i = 0
    l = np.zeros([size_Sk, ])

    if loss_generate == False:
        for a in range(r):
            i = 0
            ind = np.arange(R + a * size_Sk, R + (a + 1) * size_Sk)
            np.random.seed(a + 2019)
            np.random.shuffle(ind)
            ind = ind.astype(np.int32)
            for j in S_k:
                j = int(j)
                l[i] += loss[j - 1, ind[i]]
                i += 1
        l /= r

    if loss_generate == Bernoulli_loss:
        for j in S_k:
            j = int(j)
            random = loss_generate(mu[j - 1], r)
            l[i] = np.mean(random)
            i += 1

    if loss_generate == Gaussian_loss:
        for j in S_k:
            j = int(j)
            random = loss_generate(mu[j - 1], var[j - 1], r)
            l[i] = np.mean(random)
            i += 1

    return l


def predef_N_SH(turn_bud_to_N, size_Sk, N, K):
    if turn_bud_to_N==True:
        bud = N
        r = 0
        s = size_Sk
        for k in np.arange(np.ceil(np.log2(K))):
            r += np.floor(1.0 * N / (s * np.ceil(np.log2(K)))) * s
            s = np.ceil(1.0 * s / 2)
        N = np.ceil(1.0 * N ** 2 / r)

        while (1):
            r = 0
            s = size_Sk
            for k in np.arange(np.ceil(np.log2(K))):
                r += np.floor(1.0 * N / (s * np.ceil(np.log2(K)))) * s
                s = np.ceil(1.0 * s / 2)
            if r > bud:
                N -= 1
            else:
                break
    return N


def Successive_Halving(N, K, loss_generate, loss, mu, var, turn_bud_to_N):
    '''
    Successive-Halving algorithm for stochastic or adversarial bandit

    Inputs:
    - N: Budget;
    - K: Number of arms;
    - loss_generate: Boolean: if True, generate loss inside the function; if False, must input a loss matrix;
    - loss: Loss matrix with shape of (K, d), d should exceed the playing times;
    - mu: Means vector of stochastic losses, or vector of adversarial convergence points;
    - var: Variances vector of stochastic or adversarial losses (if neccessary).
    - turn_bud_to N: A boolean variable, if ture, the algorithm will try to fix the trial times equal or closed to the unput N.

    Outputs:
    - best arm: S_k.
    '''
    R = 0
    S_k = np.arange(1, K + 1)
    size_Sk = len(S_k)

    N = predef_N_SH(turn_bud_to_N, size_Sk, N, K)

    for k in np.arange(np.ceil(np.log2(K))):
        r = np.floor(N / (size_Sk * np.ceil(np.log2(K))))  ## pull each arm for r times
        if (r == 0):
            break
        if (r * size_Sk + R > N):
            r = np.floor(N - R) / size_Sk
        r = int(r)

        l = get_loss_SH(loss_generate, loss, mu, var, S_k, size_Sk, R, r)

        l_dict = {}  ## build a dictionary in form of {arm: loss}
        i = 0
        for j in S_k:
            l_dict[j] = l[i]
            i += 1

        R += size_Sk * r
        R = int(R)

        l_sorted = sorted(l_dict.items(), key=lambda d: d[1], reverse=False)  ## sorted on loss
        sigma = np.zeros([len(l_sorted), 1])  ## labels of arms corresponding to the ascend losses
        for i in np.arange(len(l_sorted)):
            sigma[i] = l_sorted[i][0]

        size_Sk = np.ceil(len(sigma) / 2.)  ## renew size of S_k
        size_Sk = int(size_Sk)

        S_k = sigma[np.arange(size_Sk)]  ## renew S_k
        S_k = S_k.reshape(S_k.shape[0]).astype(np.int32)

    return S_k[0]


def Ad_UCBE(c, N, K, loss_generate, loss, mu, var, turn_bud_to_N):
    '''
    Adaptive UCBE algorithm for stochastic or adversarial bandit

    Inputs:
    - c: Exploration rate;
    - N: Budget;
    - K: Number of arms;
    - loss_generate: Boolean: if True, generate loss inside the function; if False, must input a loss matrix;
    - loss: Loss matrix with shape of (K, d), d should exceed the playing times;
    - mu: Means vector of stochastic losses, or vector of adversarial convergence points;
    - var: Variances vector of stochastic or adversarial losses (if neccessary).

    Outputs:
    - best arm: S_k.
    '''
    barLog, n, t, N = predef_AdUCBE(N, K, turn_bud_to_N)

    hat_X = np.zeros([K, ])
    T = np.zeros([K, ])
    X = np.zeros([K, ])
    B = 10 ** 10 * np.ones([K, ]) ## Define B_{i,0} = +\infty

    for k in range(K - 1):
        if k == 0:
            hat_H2 = K
        else:
            sorted_hat_Delta = np.sort(max(hat_X) * np.ones([K, ]) - hat_X)
            sorted_hat_Delta[0] = sorted_hat_Delta[1]
            eva_H = np.arange(1, K + 1) / (sorted_hat_Delta ** 2 + 1e-8)
            hat_H2 = max(eva_H[(K - k):K])

        k = int(k)
        for tt in range(t[k] + 1, t[k + 1] + 1):                    
            It = int(np.argmax(B))
            T[It] += 1
            X, _ = get_reward_AdUCBE(loss_generate, loss, mu, var, X, It, tt)
            hat_X[It] = X[It] / T[It]
            B[It] = hat_X[It] + np.sqrt(c * N / (hat_H2 * T[It]))

    return np.argmax(hat_X) + 1


def S3_BA(parameters, N, K, loss_generate, loss, mu, var, turn_bud_to_N):
    '''
    Successive elimination and Successive halving for Stochastic and adversarial Best Arm

    Inputs:
    - parameters: A list of confidence and Exploration parameters used in S3_BA;
    - N: Budget;
    - K: Number of arms;
    - loss_generate: Boolean: if True, generate loss inside the function; if False, must input a loss matrix;
    - loss: Loss matrix with shape of (K, d), d should exceed the playing times;
    - mu: Means vector of stochastic losses, or vector of adversarial convergence points;
    - var: Variances vector of stochastic or adversarial losses (if neccessary).

    Outputs:
    - best arm: S_k.
    - trigger: boolean, switch to SH or not
    '''
    delta, c, C_w, C_3, C_init, C_gap = parameters

    barLog, n, t, N = predef_AdUCBE(N, K, turn_bud_to_N)  # pre-define time-related variables n, t

    B_is = 100 ** 100 * np.ones([K, ])
    hat_mu = np.zeros([K, ])  ## empirical means
    bar_mu = np.zeros([K, ])  ## unbiased estimate of empirical means
    X = np.zeros([K, ])  ## reward (cumulative)
    p = np.ones([K, ])  ## probability of choosing arms
    T = np.zeros([K, ])  ## numbers of play at current time for each arm

    ## Initial confidence bounds and widths
    lcb, bar_ucb, bar_lcb, lcb_star, width = np.zeros([K, ]), np.ones([K, ]), np.zeros([K, ]), 0, np.ones([K, ])
    gap = C_gap * width

    ## Define 'Active' and 'Bad' arms sets
    Active, Bad = list(range(0, K)), []

    ## Some simplfied definitions
    low_sum, value, hat_H2 = 0, C_w * K * c * N / delta, K
    valueK = value / hat_H2
    valueKb = 0

    trigger = False
    trigger_time = -1

    ## Begin
    for tt in range(t[0] + 1,  t[1] + 1):                
        bar_width = np.sqrt(valueK / tt)
        lcb = np.max([lcb, hat_mu - width], 0)
        bar_lcb = np.max([bar_lcb, bar_mu - bar_width], 0)
        bar_ucb = np.min([bar_ucb, bar_mu + bar_width], 0)
        lcb_star = np.max(np.max([lcb, bar_lcb], 0))

        It = np.argmax(B_is)
        T[It] += 1

        X, reward = get_reward_AdUCBE(loss_generate, loss, mu, var, X, It, tt)

        hat_mu[It] = X[It] / T[It]
        bar_mu[It] = (bar_mu[It] * (tt - 1) + reward / p[It]) / tt
        B_is[It] = hat_mu[It] + np.sqrt(c * N / (hat_H2 * T[It]))
        low_sum += lcb_star - reward
        width[It] = np.sqrt(valueK / (K * T[It]))
        gap[It] = C_gap * width[It]

    k = 1
    for tt in range(t[1] + 1, t[K - 1] + 1):                
        ## Step 1: Estimate H2 and renew 'Active' and 'Bad' arms sets
        if (tt == t[k] + 1):
            sorted_hat_Delta = np.sort(max(hat_mu) * np.ones([K, ]) - hat_mu)
            sorted_hat_Delta[0] = sorted_hat_Delta[1]
            eva_H = np.arange(1, K + 1) / (sorted_hat_Delta ** 2 + 1e-8)
            hat_H2 = max(eva_H[(K - k):K])

            k += 1
            valueK = value / hat_H2
            valueKb = C_3 * np.sqrt(valueK * N / C_w)
            active_value = C_init * c * N / (delta * hat_H2)

            B = []
            for i in Active:
                if (T[i] > active_value and hat_mu[i] + gap[i] < lcb_star):
                    B = B + [i]
            Active = list(set(Active) ^ set(B))
            Bad = Bad + B

        bar_width = np.sqrt(valueK / tt)
        lcb = np.max([lcb, hat_mu - width], 0)
        bar_lcb = np.max([bar_lcb, bar_mu - bar_width], 0)
        bar_ucb = np.min([bar_ucb, bar_mu + bar_width], 0)
        lcb_star = np.max(np.max([lcb, bar_lcb], 0))

        ## Step 2: detect non-stochastic arms
        if len(Active) == 0:
            trigger = True
            trigger_time = tt
            return np.argmax(hat_mu) + 1, trigger, trigger_time
        else:
            for i in Active:
                if (hat_mu[i] > bar_ucb[i] or hat_mu[i] < bar_lcb[i] or low_sum > valueKb):
                    trigger = True
                    trigger_time = tt
                    if ((t[-1] - tt) / (len(Active) * np.ceil(np.log2(K))) <= 1):
                        return np.argmax(hat_mu) + 1, trigger, trigger_time
                    if loss_generate == False:
                        It = Successive_Halving(t[-1] - tt, len(Active), loss_generate, loss[Active, tt:], [], [], turn_bud_to_N)
                        return Active[It - 1] + 1, trigger, trigger_time
                    if loss_generate == Bernoulli_loss:
                        It = Successive_Halving(t[-1] - tt, len(Active), loss_generate, loss[Active, tt:], mu[Active], [], turn_bud_to_N)
                        return Active[It - 1] + 1, trigger, trigger_time
                    if loss_generate == Gaussian_loss:
                        It = Successive_Halving(t[-1] - tt, len(Active), loss_generate, loss[Active, tt:], mu[Active], var[Active], turn_bud_to_N)
                        return Active[It - 1] + 1, trigger, trigger_time

        ## Step 3: draw arm
        r = np.random.rand(1)
        if ((r > delta or len(Bad) == 0) and len(Active) > 0):
            It = np.argmax(B_is[Active])
            p[It] = 1 - delta
        else:
            np.random.shuffle(Bad)
            It = Bad[0]
            p[It] = delta / len(Bad)

        T[It] += 1
        X, reward = get_reward_AdUCBE(loss_generate, loss, mu, var, X, It, tt)

        hat_mu[It] = X[It] / T[It]
        bar_mu[It] = (bar_mu[It] * (tt - 1) + reward / p[It]) / tt
        B_is[It] = hat_mu[It] + np.sqrt(c * N / (hat_H2 * T[It]))
        low_sum += lcb_star - reward
        width[It] = np.sqrt(valueK / (K * T[It]))
        gap[It] = C_gap * width[It]

    return np.argmax(hat_mu) + 1, trigger, trigger_time
