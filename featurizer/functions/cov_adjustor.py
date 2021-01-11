from functools import reduce

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


class cov_adjustor():

    def get_weighted_cov_value(self, x_df, y_df, weights):
        x_weightedmean = np.average(x_df, weights=weights, axis=0)
        y_weightedmean = np.average(y_df, weights=weights, axis=0)
        weighted_cov = np.average((x_df - x_weightedmean) * (y_df - y_weightedmean), weights=weights, axis=0)
        return weighted_cov

    def get_exponential_weight(self, window_length, decay_coef):
        w = []
        weight = 0.5 ** (1 / decay_coef)
        for i in range(window_length):
            w.append(weight ** (window_length - i - 1))
        w = [x / sum(w) for x in w]
        return w

    def Newey_West(self, factors_df, lag_period=1, decay_coef=1,forward_period=1):
        T = factors_df.shape[0]  # window_length
        names = factors_df.columns
        window_length = len(factors_df)
        w = self.get_exponential_weight(window_length, decay_coef)
        w_stats = DescrStatsW(factors_df, w)
        factors_demeaned = factors_df - w_stats.mean
        factors_demeaned = np.matrix(factors_demeaned.values)
        weighted_cov_raw = [w[t] * factors_demeaned[t].T @ factors_demeaned[t] for t in range(T)]
        weighted_cov_raw = reduce(np.add, weighted_cov_raw)
        cov_NW = weighted_cov_raw  # Newey West adjusted cov
        for i in range(1, lag_period + 1):
            w_new = [w[i + t] for t in range(T - i)]
            w_new = [x / sum(w_new) for x in w_new]
            Gammai = [w_new[i] * factors_demeaned[t].T @ factors_demeaned[i + t] for t in range(T - i)]
            Gammai = reduce(np.add, Gammai)
            cov_NW = cov_NW + forward_period * (1 - i / (1 + lag_period)) * (Gammai + Gammai.T)
        result = (pd.DataFrame(cov_NW, columns=names, index=names))
        return result

    def eigenfactor_risk_adjustment(self, Newey_West_adjustment_cov, adjust_coef):
        eigen_value, eigen_vector = np.linalg.eig(Newey_West_adjustment_cov.astype(np.float))
        eigen_value_matrix = np.diag(eigen_value)
        monte_carlo_sampling_number = 10000
        intermediate_bias_sum = np.zeros(shape=eigen_value_matrix.shape)
        for m in range(1, monte_carlo_sampling_number):  # monte_carlo
            monte_carlo_simulation = np.zeros(shape=[252, eigen_value_matrix.shape[1]])
            for i in range(Newey_West_adjustment_cov.shape[0]):
                monte_carlo_simulation[:, i] = np.random.normal(0, eigen_value[i], size=252)
            monte_carlo_cov = np.cov(monte_carlo_simulation.dot(eigen_vector), rowvar=False)
            eigen_value_m, eigen_vector_m = np.linalg.eig(monte_carlo_cov.astype(np.float))
            eigen_value_m_predicted = np.diag(eigen_value_m)
            eigen_value_m_true = (eigen_vector_m.T).dot(Newey_West_adjustment_cov).dot(eigen_vector_m)
            intermediate_bias = np.where(np.isinf(eigen_value_m_true / eigen_value_m_predicted), 0,
                                         (eigen_value_m_true / eigen_value_m_predicted))
            intermediate_bias_sum = intermediate_bias_sum + intermediate_bias
        risk_bias = np.sqrt(intermediate_bias_sum / monte_carlo_sampling_number)
        adjusted_bias = adjust_coef * (risk_bias - 1) + 1
        unbiased_eigen_value_matrix = (adjusted_bias ** 2).dot(eigen_value_matrix)
        eigenfactor_risk_adjustment_cov = eigen_vector.dot(
            unbiased_eigen_value_matrix.dot(eigen_value_matrix).dot(eigen_vector.T))
        return eigenfactor_risk_adjustment_cov

    def volatility_regime_adjustment(self, eigenfactor_risk_adjustment_cov, factors_df, decay_coef):
        window_length = len(factors_df)
        w = self.get_exponential_weight(window_length, decay_coef)
        empirical_factor_volitality = pd.Series(data=np.sqrt(np.diag(factors_df.cov())),
                                                index=factors_df.columns)
        bias = pd.Series(index=factors_df.index)
        for date in factors_df.index.tolist():
            bias.loc[date] = np.square(factors_df.loc[date] / empirical_factor_volitality).sum() / len(
                factors_df.columns)
        w = np.array(w)
        lambda_f = np.sqrt(w.dot(bias))
        volatility_regime_adjustment_cov = lambda_f ** (2) * eigenfactor_risk_adjustment_cov
        return volatility_regime_adjustment_cov
