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

import torch
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from featurizer.functions.algebra_statistic import weighted_average, weighted_std, downside_std

# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def rolling_sum(tensor, window=1, dim=0):
    ret = torch.cumsum(tensor, dim=dim)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window-1]= float("nan")
    return ret

def rolling_sum_(tensor, window=1, dim=0):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).sum()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_sum3d(tensor, window=1, dim=1):
    ret = torch.cumsum(tensor, dim=dim)
    ret[:,window:] = ret[:,window:] - ret[:,:-window]
    ret[:,:window-1]= float("nan")
    return ret

def rolling_mean(tensor, window=1):
    ret = torch.cumsum(tensor, dim=0)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window-1]= float("nan")
    return ret/window

def rolling_mean_(tensor, window=1):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).mean()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_weighted_mean(tensor, window=1, halflife=90):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(lambda x: weighted_average(x,halflife=halflife))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(tensor, window=1):
    if window == 0:
        return tensor
    
    e = torch.empty_like(tensor, dtype=tensor.dtype, device=tensor.device)
    if window > 0:
        e[:window] = float("nan")
        e[window:] = tensor[:-window]
    else:
        e[window:] = float("nan")
        e[:window] = tensor[-window:]
    return e

def diff(tensor, period=1):
    shiftd_tensor = shift(tensor, window=period)
    diff = tensor - shiftd_tensor
    return diff

def pct_change(tensor, period=1):
    shiftd_tensor = shift(tensor, window=period)
    diff = tensor - shiftd_tensor
    output = diff.div(shiftd_tensor)
    return output 

#https://stackoverflow.com/questions/54564253/how-to-calculate-the-cumulative-product-of-a-rolling-window-in-pandas
def rolling_prod(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(np.prod)
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_var(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).var()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor 

def rolling_std(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).std()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_weighted_std(tensor, window, halflife=90):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).apply(lambda x: weighted_std(x, halflife=halflife))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_downside_std(tensor, tensor_benchmark, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_benchmark_np = tensor_benchmark.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    #tensor_benchmark_s = pd.Series(tensor_benchmark_np)
    output_df = tensor_df.rolling(window).apply(lambda x: downside_std(x, tensor_benchmark_np))
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_skew(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).skew()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_max(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).max()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_corr(tensor_x, tensor_y, window):
    tensor_x_np = tensor_x.cpu().detach().numpy()
    tensor_y_np = tensor_y.cpu().detach().numpy()
    tensor_x_df = pd.DataFrame(tensor_x_np)
    tensor_y_df = pd.DataFrame(tensor_y_np)
    output_df = tensor_x_df.rolling(window).corr(tensor_y_df)
    output_tensor = torch.tensor(output_df.values, dtype=tensor_x.dtype, device=tensor_x.device)
    return output_tensor

def rolling_cov(tensor_x, tensor_y, window, trailing_window=0):
    tensor_x_np = tensor_x.cpu().detach().numpy()
    tensor_y_np = tensor_y.cpu().detach().numpy()
    tensor_x_df = pd.DataFrame(tensor_x_np)
    tensor_y_df = pd.DataFrame(tensor_y_np)
    output_df = tensor_x_df.rolling(window).cov(tensor_y_df)
    output_tensor = torch.tensor(output_df.values, dtype=tensor_x.dtype, device=tensor_x.device)
    return output_tensor 

def ema(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.ewm(span=window,min_periods=window).mean()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor


def rolling_min(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).min()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor 

def rolling_max(tensor, window):
    tensor_np = tensor.cpu().detach().numpy()
    tensor_df = pd.DataFrame(tensor_np)
    output_df = tensor_df.rolling(window).max()
    output_tensor = torch.tensor(output_df.values, dtype=tensor.dtype, device=tensor.device)
    return output_tensor

def rolling_rank(np_data):
    return rankdata(np_data)[-1]

def ts_rank(data_ts, window=10):
    data_df=pd.DataFrame(data_ts)
    output_df = data_df.rolling(window).apply(rolling_rank, raw=False)
    output_np= np.array(output_df)
    output_tensor=torch.tensor(output_np)
    return output_tensor

def rank(data_ts, axis=1, pct=True):
    data_df=pd.DataFrame(data_ts)
    output_df = data_df.rank(axis=axis, pct=pct)
    output_np = np.array(output_df)
    output_tensor = torch.tensor(output_np)
    return output_tensor

def rolling_scale(data_ts, window=10, k=1):
    output_tensor = k*data_ts/rolling_sum_(torch.abs(data_ts), window=window)
    return output_tensor

print(666)