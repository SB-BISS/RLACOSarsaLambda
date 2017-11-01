

from sklearn.mixture import GMM
import pylab as pl
from scipy import linalg
import numpy as np

def gmm_map_qb(X, gmm, rho=.1, epsilon=1, niter=10):
    """
       GMM adaptation using Quasi-Bayes MAP method.
       Usage: gmm_map_qb(X, gmm, rho, epsilon, niter)
       //Ref. 1) (1994 Gauvain, Lee) Maximum a Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains
       //Ref. 2) (1997 Huo, Lee) On-line adaptive learning of the continuous density hidden markov model based on approximate recursive bayes estimate
       //Ref. 3) (2010 Kim, Loizou) Improving Speech Intelligibility in Noise Using Environment-Optimized Algorithms"""

    # init
    logprob, pcompx = gmm.eval(X)
    psum = np.sum(pcompx, axis=0)  # (18)
    # remove illed gaussians
    ill_g = (psum == 0);
    if any(ill_g):
        valid = psum > 0
        gmm.means_ = gmm.means_[valid, :]
        gmm.weights_ = gmm.weights_[valid]
        gmm.weights_ = gmm.weights_ / sum(gmm.weights_)
        gmm.covars_ = gmm.covars_[valid]

    logprob, pcompx = gmm.eval(X)
    psum = np.sum(pcompx, axis=0)  # (18)
    K, nDim = gmm.means_.shape
    tau = psum * epsilon  # (22)
    tau_update = tau
    nu = 1 + tau  # (23)
    nu_update = nu
    alpha = nDim + tau  # (24)
    alpha_update = alpha
    mu = np.empty([K, nDim])
    mu_update = np.empty([K, nDim])
    yu = np.empty([K, nDim, nDim])
    yu_update = np.empty([K, nDim, nDim])
    for k in range(0, K):
        mu[k] = gmm.means_[k]  # (25)
        yu[k] = tau[k] * gmm.covars_[k]  # (26)

    # EM iterations
    s = np.empty([K, nDim, nDim])
    N = X.shape[0]
    for iter in range(0, niter):
        print 'iter=', iter
        print np.sum(gmm.weights_)
        # plotgmm(gmm,X)
        # E-step: posterior probabilities
        logprob, pcompx = gmm.eval(X)

        # remove illed gaussians
        psum = np.sum(pcompx, axis=0)  # (18)
        # print psum
        # raw_input()
        ill_g = (psum == 0);
        # print ill_g
        if any(ill_g):
            valid = psum > 0
            gmm.means_ = gmm.means_[valid, :]
            gmm.weights_ = gmm.weights_[valid]
            gmm.weights_ = gmm.weights_ / sum(gmm.weights_)
            gmm.covars_ = gmm.covars_[valid]
            mu = mu[valid]
            mu_update = mu_update[valid]
            yu = yu[valid]
            yu_update = yu_update[valid]
            tau = tau[valid]
            tau_update = tau_update[valid]
            alpha = alpha[valid]
            alpha_update = alpha_update[valid]
            nu = nu[valid]
            nu_update = nu_update[valid]
            K = gmm.means_.shape[0]
            continue

        # M-step, eqs. from KimLoizou'10
        # Hyper-parameters
        psum = np.sum(pcompx, axis=0)  # (18)
        # print np.sum(pcompx,axis=1)
        # print 'psum',psum
        # print gmm.weights_
        # raw_input()
        x_expected = np.dot(pcompx.T, X) / np.tile(psum[np.newaxis].T, (1, nDim))  # (19)
        # print 'x_expected',x_expected
        for k in range(0, K):
            # raw_input()
            # (20)
            s[k] = np.dot((X - np.tile(x_expected[k], (N, 1))).T,
                          (X - np.tile(x_expected[k], (N, 1))) * np.tile(pcompx[:, k][np.newaxis].T, (1, nDim)))
            # (15)
            # print 'yu[k]',yu[k]
            # print 'x_expected - mu', (x_expected-mu).T
            # yu_update[k] = rho*yu[k] + s[k] + rho*tau[k]*psum[k]/(rho*tau[k]+psum[k])*np.dot((x_expected-mu).T, x_expected-mu)
            yu[k] = rho * yu[k] + s[k] + rho * tau[k] * psum[k] / (rho * tau[k] + psum[k]) * np.dot(
                (x_expected[k, :] - mu[k, :]).T, x_expected[k, :] - mu[k, :])
            # print 'yu[k],after',yu[k]

        # (21)
        beta = psum / (rho * tau + psum)
        # mu, eq (14)
        # print beta.shape
        # print mu.shape
        # print x_expected.shape
        # mu_update = np.tile(beta[np.newaxis].T,(1,nDim))*x_expected + np.tile((1-beta)[np.newaxis].T,(1,nDim))*mu
        mu = np.tile(beta[np.newaxis].T, (1, nDim)) * x_expected + np.tile((1 - beta)[np.newaxis].T, (1, nDim)) * mu
        # tau, eq (11)
        # tau_update = rho*tau + psum
        tau = rho * tau + psum
        # alpha, eq (12)
        # alpha_update = rho*(alpha-nDim) + nDim + psum
        alpha = rho * (alpha - nDim) + nDim + psum
        # print 'alpha',alpha
        # nu, eq (13)
        # nu_update = rho*(nu-1) + 1 + psum
        nu = rho * (nu - 1) + 1 + psum

        # GMM parameters
        # weight, (27)
        # gmm.weights_ = (nu_update-1)/np.sum(nu_update-1)
        gmm.weights_ = (nu - 1) / np.sum(nu - 1)
        ill = (gmm.weights_ == 0)
        print np.sum(gmm.weights_)
        # mean, (28)
        # gmm.means_ = mu_update
        gmm.means_ = mu
        # sigma, (29)
        for k in range(0, K):
            # if alpha_update[k] != nDim:
            if alpha[k] != nDim:
                # gmm.covars_[k] = yu_update[k]/(alpha_update[k]-nDim)
                gmm.covars_[k] = yu[k] / (alpha[k] - nDim)
            else:
                # gmm.covars_[k] = yu_update[k]/tau_update[k]
                gmm.covars_[k] = yu[k] / tau[k]
            try:
                np.linalg.cholesky(gmm.covars_[k])
            except:
                ill[k] = 1
                print 'cov_%d not positive definite' % k

        # remove non positive definite matrices
        if np.any(ill):
            valid = (ill == 0)
            gmm.means_ = gmm.means_[valid]
            gmm.weights_ = gmm.weights_[valid]
            gmm.weights_ = gmm.weights_ / sum(gmm.weights_)
            gmm.covars_ = gmm.covars_[valid]
            mu = mu[valid]
            mu_update = mu_update[valid]
            yu = yu[valid]
            yu_update = yu_update[valid]
            tau = tau[valid]
            tau_update = tau_update[valid]
            alpha = alpha[valid]
            alpha_update = alpha_update[valid]
            nu = nu[valid]
            nu_update = nu_update[valid]
            K = gmm.means_.shape[0]


    return gmm