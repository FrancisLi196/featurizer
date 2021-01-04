import numpy as np
import pandas as pd
import statsmodels.api as sm


class betacalculator():

    def beta_normal(self, return_df, market_df):
        ols = sm.OLS(return_df, market_df).fit()
        beta = ols.params[0]
        return beta

    def beta_EMVA(self, return_df, market_df, decay_coef):
        weights = []
        weight = 0.5 ** (1 / decay_coef)
        for i in range(len(market_df)):

            if pd.isna(return_df.iloc[i].item()):
                weights.append(0)
            else:
                weights.append(weight ** (len(market_df) - i - 1))
        weights = [x / sum(weights) for x in weights]
        wls = sm.WLS(return_df, market_df, weights=weights).fit()
        beta = wls.params[0]
        return beta

    def beta_shrink(self, return_df, market_df, decay_coef, beta_base_df):
        beta_base_std = np.std(beta_base_df)
        beta_base = np.mean(beta_base_df)
        weights = []
        weight = 0.5 ** (1 / decay_coef)
        for i in range(len(market_df)):

            if pd.isna(return_df.iloc[i].item()):
                weights.append(0)
            else:
                weights.append(weight ** (len(market_df) - i - 1))
        weights = [x / sum(weights) for x in weights]
        wls = sm.WLS(return_df, market_df, weights=weights).fit()
        beta_hist = wls.params[0]
        beta_hist_std = wls.bse[0]
        adjust_coef = (beta_base_std ** 2) / (beta_base_std ** 2 + beta_hist_std ** 2)
        beta_shrinked = adjust_coef * beta_hist + (1 - adjust_coef) * beta_base
        return beta_shrinked


