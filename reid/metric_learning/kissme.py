from __future__ import absolute_import

import numpy as np
from metric_learn.base_metric import BaseMetricLearner


def validate_cov_matrix(M):
    M = (M + M.T) * 0.5
    k = 0
    I = np.eye(M.shape[0])
    while True:
        try:
            _ = np.linalg.cholesky(M)
            break
        except np.linalg.linalg.LinAlgError:
            # Find the nearest positive definite matrix for M. Modified from
            # http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            # Might take several minutes
            print('Find the nearest positive definite matrix, k = ', k)
            k += 1
            w, v = np.linalg.eig(M)
            min_eig = v.min()
            M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M


class KISSME(BaseMetricLearner):
    def __init__(self):
        self.M_ = None

    def metric(self):
        return self.M_
        
    def fit_Crossview(self, galX, probX, galLabels, probLabels):
        num_gal = galX.shape[0]
        num_prob = probX.shape[0]
        X_gal, X_prob = np.meshgrid(np.arange(num_gal), np.arange(num_prob))
        X_gal = X_gal.flatten()
        X_prob = X_prob.flatten()
        print('#all pairs:', len(X_gal))
        matches = (galLabels[X_gal] == probLabels[X_prob])
        num_matches = matches.sum()
        num_non_matches = len(matches) - num_matches
        print('num_matches:', num_matches)
        idx_gal = X_gal[matches]
        idx_prob = X_prob[matches]
        S = galX[idx_gal] - probX[idx_prob]
        C1 = S.transpose().dot(S) / num_matches # C1 is the covariance matrix of matched sample pairs
        p = np.random.choice(num_non_matches, num_matches, replace=False)
        idx_gal = X_gal[~matches]
        idx_prob = X_prob[~matches]
        idx_gal = idx_gal[p]
        idx_prob = idx_prob[p]
        S = galX[idx_gal] - probX[idx_prob]
        C0 = S.transpose().dot(S) / num_matches # C0 is the covariance matrix of unmatched sample pairs
        self.M_ = np.linalg.inv(C1) - np.linalg.inv(C0)
        # self.M_ = validate_cov_matrix(self.M_)
        
    def fit(self, X, y=None):
        n = X.shape[0]
        if y is None:
            y = np.arange(n)
        X1, X2 = np.meshgrid(np.arange(n), np.arange(n))
        X1, X2 = X1[X1 < X2], X2[X1 < X2]
        print('#all pairs:', len(X1))
        matches = (y[X1] == y[X2])
        num_matches = matches.sum()
        num_non_matches = len(matches) - num_matches
        print('num_matches:', num_matches)
        idxa = X1[matches]
        idxb = X2[matches]
        S = X[idxa] - X[idxb]
        C1 = S.transpose().dot(S) / num_matches # C1 is the covariance matrix of matched sample pairs
        p = np.random.choice(num_non_matches, num_matches, replace=False)
        idxa = X1[~matches]
        idxb = X2[~matches]
        idxa = idxa[p]
        idxb = idxb[p]
        S = X[idxa] - X[idxb]
        C0 = S.transpose().dot(S) / num_matches # C0 is the covariance matrix of unmatched sample pairs
        self.M_ = np.linalg.inv(C1) - np.linalg.inv(C0)
        # self.M_ = validate_cov_matrix(self.M_)
        self.X_ = X
