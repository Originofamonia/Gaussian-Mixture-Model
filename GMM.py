# https://github.com/mijungi/dpem_code
import numpy as np
from scipy.optimize import fsolve
import scipy.stats as sp


class GaussianMixModel(object):
    def __init__(self, x, total_eps=0.1, k=2):
        # Algorithm can work for any number of columns in dataset
        x = np.asarray(x)
        self.n, self.d = x.shape  # num of data points, dimension
        self.data = x.copy()
        # number of mixtures
        self.k = k
        self.thresh = 1e-4
        self.max_iter = 10
        self.lap = False
        self.total_del = 1e-4
        self.total_eps = total_eps
        self.pi_prior = 2 * np.ones(self.k)

    def _init(self):
        # init mixture means/sigmas
        self.mu = np.asmatrix(np.random.random((self.k, self.d)) + np.mean(self.data))  # [k, d]
        self.sigma = np.array([np.asmatrix(np.identity(self.d)) for i in range(self.k)])  # [k, d, d]
        self.pi = np.ones(self.k) / self.k  # proportion of each cluster [k]
        # z Latent Variable giving probability of each point for each distribution
        self.z = np.asarray(np.empty((self.n, self.k), dtype=float))  # [n, k]

        if self.lap:
            delta_i = self.total_del / (self.max_iter * self.k)
        else:
            delta_i = self.total_del / (self.max_iter * (self.k + 1))
        self.c2 = 2 * np.log(1.25 / delta_i)
        self.eps_prime = self.total_eps / (self.max_iter * (2 * self.k + 1))

    def fit(self, ):
        # Algorithm will run until max of log-likelihood is achieved
        self._init()
        num_iters = 0
        logl = 1
        previous_logl = 0
        while num_iters < self.max_iter:
            # previous_logl = self.loglikelihood()
            if self.total_eps != 0:
                ess = self.e_step()
                self.m_step(ess)
            else:
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
                tmp += sp.multivariate_normal.pdf(self.data[i, :], self.mu[j, :].A1, self.sigma[j, :]) \
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
                                                 self.mu[j].A1,
                                                 self.sigma[j]) * self.pi[j]

                den += num

                self.z[i, j] = num
            self.z[i, :] /= den
            # assert self.z[i, :].sum() - 1 < self.thresh  # Program stops if this condition is false

        if self.total_eps != 0:  # DP version
            ess = self.cond_gauss_cpd_compute_ess()
            return ess

    def cond_gauss_cpd_compute_ess(self, ):
        """
        Compute the expected sufficient statistics for a condGaussCpd
        input: data, z (marginal probability of the discrete parent for each observation)
        return: xbar, xx, wsum, 2ndmoment
        """
        wsum = np.sum(self.z, axis=0)
        wsum = np.clip(wsum, a_min=1.01, a_max=None)  # [k]
        xbar = np.dot(self.data.T, self.z) / wsum  # [d, k]
        xx = np.zeros((self.k, self.d, self.d))  # [k, d, d] to match sigma's dim
        second_moment = np.zeros((self.k, self.d, self.d))  # [k, d, d]

        for j in range(self.k):
            xc = self.data - xbar[:, j]  # xc: [n, d], the same dim as data
            second_moment[j] = np.dot(self.data.T * self.z[:, j], self.data)
            xx[j] = np.dot(xc.T * self.z[:, j], xc)

        return xbar, xx, wsum, second_moment

    def m_step(self, ess=None):

        # Updating mean and variance
        if self.total_eps != 0:  # DP version
            self.cond_gauss_cpd_fit_ess(ess)
            self.pi = (ess.wsum + self.pi_prior - 1) / np.sum(ess.wsum + self.pi_prior - 1)

            if self.lap:
                lap_noise_var = 2 / (self.n * self.eps_prime)
                noise = self.laprnd(self.k, 1, 0, np.sqrt(2) * lap_noise_var)
            else:
                sensitiv = 2 / self.n
                sigma = sensitiv / self.eps_prime
                noise = np.random.multivariate_normal(np.zeros(self.k), self.c2 * sigma ** 2 * np.eye(self.k))

            noised_pi = self.pi + noise
            noised_pi = np.clip(noised_pi, 0, 1)
            noised_numerator = noised_pi / np.sum(noised_pi) * self.n
            self.pi = (noised_numerator + self.pi_prior - 1) / np.sum(noised_numerator + self.pi_prior - 1)
        else:  # non-DP version
            for j in range(self.k):
                const = self.z[:, j].sum()
                self.pi[j] = 1 / self.n * const
                mu_j = np.zeros(self.d)
                sigma_j = np.zeros((self.d, self.d))
                for i in range(self.n):
                    mu_j += (self.data[i, :] * self.z[i, j])
                    sigma_j += self.z[i, j] * (
                            (self.data[i, :] - self.mu[j, :]).T * (self.data[i, :] - self.mu[j, :]))

                self.mu[j] = mu_j / const
                self.sigma[j] = sigma_j / const

    def cond_gauss_cpd_fit_ess(self, ess):
        xbar, xx, wsum, second_moment = ess
        kappa0 = 1.01
        m0 = np.zeros(self.d)
        nu0 = self.d + 1
        s0 = 0.1 * np.eye(self.d)
        # mu = np.zeros((self.k, self.d))
        # sigma = np.zeros((self.k, self.d, self.d))

        # sensitivity for mean
        nk_plus_kappa0 = wsum + kappa0
        l1sen_mean = 2 * np.sqrt(self.d) * (1 / nk_plus_kappa0)
        const = nu0 + self.d + 2
        nk_plus_const = wsum + const
        sen_cov = 1 / nk_plus_const

        for k in range(self.k):
            xbark = xbar[:, k]
            xxk = xx[k]
            wk = wsum[k]

            # add noise to mean
            mn = (wk * xbark + kappa0 * m0) / (wk + kappa0)

            if self.lap:
                noise_var_for_mean2 = l1sen_mean[k] / self.eps_prime
                noise_for_mean = self.laprnd(self.d, 1, 0, np.sqrt(2) * noise_var_for_mean2)
            else:  # Gaussian
                mudiff = l1sen_mean[k] / np.sqrt(self.d)
                sigma = mudiff / self.eps_prime
                noise_for_mean = np.random.multivariate_normal(np.zeros(self.d),
                                                               self.c2 * sigma ** 2 * np.eye(self.d)).T
            new_mu = mn + noise_for_mean  # don't understand why mn, why not self.mu[k]
            if np.linalg.norm(new_mu) > 1:
                new_mu = new_mu / np.linalg.norm(new_mu)
            self.mu[k] = new_mu

            # add noise to sigma
            a = (kappa0 * wk) / (kappa0 + wk)
            b = nu0 + wk + self.d + 2
            s_prior = np.dot(np.expand_dims((xbark - m0), axis=1), np.expand_dims((xbark - m0), axis=0))
            sigma_org = (s0 + xxk + a * s_prior) / b  # why it's calculated, not self.sigma[k]

            # Gaussian noise
            sigma = sen_cov[k] / self.eps_prime
            noisemat = np.triu(np.random.normal(0, np.sqrt(self.c2) * sigma, size=(self.d, self.d)))
            z = noisemat + np.tril(noisemat.T, -1)
            noised_sigma = sigma_org + z

            # make sure the matrix is pd
            d, v = np.linalg.eig(noised_sigma)  # d: eigenvalues (vec), v: eigenvectors; satisfies A.v = v.diag_d
            tol = 1e-3
            d[d < tol] = tol
            diag_d = np.zeros((len(d), len(d)))  # square diag matrix
            np.fill_diagonal(diag_d, d)
            new_noised_sigma = np.dot(np.dot(v, diag_d), v.T)

            self.sigma[k] = new_noised_sigma

    def laprnd(self, m, n, mu, sigma):
        """
        generate i.i.d. laplacian random number drawn from laplacian distribution
        mu: mean
        sigma: standard deviation
        [m, n]: the dimension of return
        """
        u = np.random.rand(m, n) - 0.5
        b = sigma / np.sqrt(2)
        y = mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u))
        return y
