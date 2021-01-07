# -*- coding: utf-8 -*-
# Copyright StateOfTheArt.quant. 
#
# * Commercial Usage: please contact allen.across@gmail.com
# * Non-Commercial Usage:
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

def create_weight_by_halflife(n, halflife=4):
    lambda_ = 0.5**(1/halflife)
    w = np.array([lambda_**n for n in range(n)][::-1])
    w = w/w.sum()
    return w

def weighted_average(matrix, axis=0, halflife=90):
    """Return the weighted average"""
    Tn = matrix.shape[axis]   # number of Time period
    w = create_weight_by_halflife(n=Tn, halflife=halflife)
    # wighted average of attributes
    average = np.average(matrix, axis=axis, weights=w)
    return average

def weighted_std(matrix, axis=0, halflife=90):
    """Return the weighted standard deviation""" # attention have a little bias /n or /(n-1)
    Tn = matrix.shape[axis]   # number of Time period
    w = create_weight_by_halflife(n=Tn, halflife=halflife)
    
    average = np.average(matrix, axis=axis, weights=w)
    variance = np.average((matrix-average)**2, axis=axis, weights=w) # Fast and numerically precise
    return np.sqrt(variance)

def weighted_std_from_stats(matrix, axis=0, halflife=90):
    Tn = matrix.shape[axis]   # number of Time period
    w = create_weight_by_halflife(n=Tn, halflife=halflife)

    weighted_stats = DescrStatsW(matrix, weights=w, ddof=0)
    return weighted_stats.std

def downside_std(diff_arr):
    indicator = diff_arr<0
    diff_squared = np.power(diff_arr, 2)
    diff_squared_under_indicator = diff_squared * indicator
    mean_squared_sum = sum(diff_squared_under_indicator)/ (len(diff_arr)-1) 
    return np.power(mean_squared_sum, 0.5)

def upside_std(diff_arr):
    indicator = diff_arr>0
    diff_squared = np.power(diff_arr, 2)
    diff_squared_under_indicator = diff_squared * indicator
    mean_squared_sum = sum(diff_squared_under_indicator)/ (len(diff_arr)-1) 
    return np.power(mean_squared_sum, 0.5)



if __name__ == "__main__":
    import pandas as pd
    
    data = np.array([[1,2,3,4,5,6],[3,4,5,6,7,8]]).transpose()
    data_df = pd.DataFrame(data)
    #average = weighted_average(data, axis=0)
    #std = weighted_std(data, axis=0)
    #std_stats = weighted_std_from_stats(data)
    #df = pd.DataFrame(data)
    #std_df = df.ewm(halflife=90).std()
    
    data_arr = np.array([1,-2,3,-4,5,6])
    data_benchmark = np.array([2,-3, 2,-3,4,9])
    
    downside_volatility_value = downside_std(data_arr,data_benchmark)
    
    downside_volatility_batch = data_df.apply(lambda x: downside_std(x, data_benchmark))