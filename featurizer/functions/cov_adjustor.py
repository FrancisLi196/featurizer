import numpy as np
import pandas as pd


class cov_adjustor():

    def get_weighted_cov_value(self, x_df, y_df, weights):
        x_weightedmean = np.average(x_df, weights=weights, axis=0)
        y_weightedmean = np.average(y_df, weights=weights, axis=0)
        weighted_cov = np.average((x_df - x_weightedmean) * (y_df - y_weightedmean), weights=weights, axis=0)
        return weighted_cov

    def Newey_West_cov(self, factors_df, decay_coef, lag_length, prediction_period=21):
        window_length = len(factors_df)
        w = []
        weight = 0.5 ** (1 / decay_coef)
        for i in range(window_length):
            w.append(weight ** (window_length - i - 1))
        cov_raw = np.cov(factors_df, rowvar=False, aweights=w)
        cov_NW = cov_raw.copy()
        for i in range(cov_raw.shape[0]):
            for j in range(cov_raw.shape[1]):
                temp_result = 0  # cplus + cminus
                for delta in range(1, lag_length + 1):
                    temp_length = window_length - delta
                    temp_weights = []
                    for k in range(temp_length):
                        temp_weights.append(weight ** (temp_length - k - 1))
                    temp_x_minus = factors_df[delta:][i].reset_index(drop=True)
                    temp_y_minus = factors_df[0:window_length - delta][j].reset_index(drop=True)
                    c_minus = self.get_weighted_cov_value(temp_x_minus, temp_y_minus, weights=temp_weights)
                    temp_x_plus = factors_df[0:window_length - delta][i].reset_index(drop=True)
                    temp_y_plus = factors_df[delta:][j].reset_index(drop=True)
                    c_plus = self.get_weighted_cov_value(temp_x_plus, temp_y_plus, weights=temp_weights)
                    temp_result = temp_result + (1 - (delta / (lag_length + 1))) * (c_plus + c_minus)
                cov_NW[i][j] = prediction_period * (cov_raw[i][j] + temp_result)
        return cov_NW

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

    def volatility_regime_adjustment(self, eigenfactor_risk_adjustment_cov, factors_df, decay_coef ):
        window_length = len(factors_df)
        w = []
        weight = 0.5 ** (1 / decay_coef)
        for i in range(window_length):
            w.append(weight ** (window_length - i - 1))
        empirical_factor_volitality = pd.Series(data=np.sqrt(np.diag(factors_df.cov())),
                                                index=factors_df.columns)
        bias = pd.Series(index=factors_df.index)
        for date in factors_df.index.tolist():
            bias.loc[date] = np.square(factors_df.loc[date] / empirical_factor_volitality).sum() / len(
                factors_df.columns)
        w = np.array(w)
        lambda_f = np.sqrt(w.dot(bias) / w.sum())
        volatility_regime_adjustment_cov = lambda_f ** (2) * eigenfactor_risk_adjustment_cov
        return volatility_regime_adjustment_cov


if __name__ == '__main__':
    a = cov_adjustor()
    x_test_df = pd.DataFrame([0, 1, 2])
    y_test_df = pd.DataFrame([0, 1, 7])
    x_test_df_last = pd.DataFrame([-1, -2, -3])
    y_test_df_last = pd.DataFrame([4, 5, 6])
    combined = pd.DataFrame([[0, 2], [1, 6], [2, 0], [7, 9]])
    previous = pd.DataFrame([[-1, -2], [-3, -6], [-2, -4]])
    fakefactor = pd.DataFrame(data=[[0.5,0.6,0.7],[0.1,0.2,0.3],[1,2,3]],index=['20120101','20120202','20120205'])
    a.volatility_regime_adjustment(pd.DataFrame([[4, 0,1], [0, 2,0],[1,0,1]]),fakefactor, 1)
