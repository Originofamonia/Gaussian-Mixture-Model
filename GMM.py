import numpy as np

import scipy.stats as sp


class GaussianMixModel(object):
    def __init__(self, x, k=2):
        # Algorithm can work for any number of columns in dataset
        x = np.asarray(x)
        self.n, self.d = x.shape  # num of data points, dimension
        self.data = x.copy()
        print(np.mean(x))
        # number of mixtures
        self.k = k
        self.thresh = 1e-4

    def _init(self):
        # init mixture means/sigmas
        self.mean = np.asmatrix(np.random.random((self.k, self.d)) + np.mean(self.data))
        self.sigma = np.array([np.asmatrix(np.identity(self.d)) for i in range(self.k)])
        self.pi = np.ones(self.k) / self.k  # proportion of each cluster
        # Z Latent Variable giving probability of each point for each distribution
        self.Z = np.asmatrix(np.empty((self.n, self.k), dtype=float))

    def fit(self,):
        # Algorithm will run until max of log-likelihood is achieved
        self._init()
        num_iters = 0
        logl = 1
        previous_logl = 0
        while logl - previous_logl > self.thresh:
            previous_logl = self.loglikelihood()
            self.e_step()
            self.m_step()
            num_iters += 1
            logl = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f' % (num_iters, logl))
        print('Terminate at %d-th iteration:log-likelihood is %.6f' % (num_iters, logl))

    def loglikelihood(self):
        logl = 0
        for i in range(self.n):
            tmp = 0
            for j in range(self.k):
                # print(self.sigma_arr[j])
                tmp += sp.multivariate_normal.pdf(self.data[i, :], self.mean[j, :].A1, self.sigma[j, :]) \
                       * self.pi[j]
            logl += np.log(tmp)
        return logl

    def e_step(self):
        # Finding probability of each point belonging to each pdf and putting it in latent variable Z
        for i in range(self.n):
            den = 0
            for j in range(self.k):
                # prob of data[i] belongs to mu[j], sigma[j]
                num = sp.multivariate_normal.pdf(self.data[i, :],
                                                 self.mean[j].A1,
                                                 self.sigma[j]) * self.pi[j]

                den += num

                self.Z[i, j] = num
            self.Z[i, :] /= den
            assert self.Z[i, :].sum() - 1 < self.thresh  # Program stop if this condition is false

    def m_step(self):
        # Updating mean and variance
        for j in range(self.k):
            const = self.Z[:, j].sum()
            self.pi[j] = 1 / self.n * const
            mu_j = np.zeros(self.d)
            sigma_j = np.zeros((self.d, self.d))
            for i in range(self.n):
                mu_j += (self.data[i, :] * self.Z[i, j])
                sigma_j += self.Z[i, j] * (
                        (self.data[i, :] - self.mean[j, :]).T * (self.data[i, :] - self.mean[j, :]))

            self.mean[j] = mu_j / const
            self.sigma[j] = sigma_j / const
