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
import pandas as pd
import numpy as np
from functools import reduce
import pdb

'''
Please see 'unittest_split.py' and 'create_expected_output_split.py' for comprehensive examples for functions in this module.
'''

def split(tensor, window=5, step=1, offset=0, keep_tail=True):
    """
    :param tensor: 2d numpy or 2d tensor
    :param window: int, size of window default=5
    :param step: int, size between two windows default=1
    :param offset: int, first window offset default=0
    :param keep_tail: Boolean , {True : save tail of data,; False : possible not save tail of data} default True
    :return: list within numpy data
    Examples::
        (note: the example data is not in the correct input format; it's intended for understanding only)
        >>> data = np.array([1,2,3,4,5,6,7,8,9,10])
        >>> # keep_tail is True
        >>> split_list = split(data, window=4, step=5, offset=0, keep_tail=True)
        >>> split_list  # [array([1]), array([2, 3, 4, 5]), array([ 7,  8,  9, 10])]
        >>> # keep_tail is False
        >>> split_list = split(data, window=4, step=5, offset=0, keep_tail=False)
        >>> split_list # [array([1, 2, 3, 4]), array([6, 7, 8, 9]), array([10])]
    """
    window, step, offset = int(window), int(step), int(offset)
    sample_list = []
    index = int((len(tensor) - window - offset) / step) + 1 #total steps
    remain = int(len(tensor) - window - offset - (index - 1) * step)
    #print('remain : ', remain)
    
    if keep_tail:
        start_index = remain+offset#
        if remain > 0:
            sample_list.append(tensor[offset:offset+remain])
        for i in range(index):
            window_data = tensor[start_index + i * step : start_index + window + i * step]
            sample_list.append(window_data)
    else:
        start_index = offset
        for i in range(index):
            window_data = tensor[start_index + i * step : start_index + window + i * step]
            sample_list.append(window_data)
        if remain > 0:
            sample_list.append(tensor[-remain:])

    return sample_list

def split3d(tensor, window=5, step=1, offset=0, keep_tail=True, dim=1):
    """
    :param tensor: 3d numpy or 3d tensor
    :param window: int, size of window default=5
    :param step: int, size between two windows default=1
    :param offset: int, first window offset default=0
    :param keep_tail: Boolean , {True : save tail of data,; False : possible not save tail of data} default True
    :return: list within numpy data
    Examples::
        (note: the example data is not in the correct input format; it's intended for understanding only)
        >>> data = np.array([1,2,3,4,5,6,7,8,9,10])
        >>> # keep_tail is True
        >>> split_list = split(data, window=4, step=5, offset=0, keep_tail=True)
        >>> split_list  # [array([1]), array([2, 3, 4, 5]), array([ 7,  8,  9, 10])]
        >>> # keep_tail is False
        >>> split_list = split(data, window=4, step=5, offset=0, keep_tail=False)
        >>> split_list # [array([1, 2, 3, 4]), array([6, 7, 8, 9]), array([10])]
    """
    window, step, offset = int(window), int(step), int(offset)
    sample_list = []
    length = tensor.shape[dim]
    index = int((length - window - offset) / step) + 1 #total steps
    remain = int(length - window - offset - (index - 1) * step)
    #print('remain : ', remain)
    
    if keep_tail:
        start_index = remain+offset#
        if remain > 0:
            sample_list.append(tensor[:,offset:offset+remain])
        for i in range(index):
            window_data = tensor[:,start_index + i * step : start_index + window + i * step]
            sample_list.append(window_data)
    else:
        start_index = offset
        for i in range(index):
            window_data = tensor[:,start_index + i * step : start_index + window + i * step]
            sample_list.append(window_data)
        if remain > 0:
            sample_list.append(tensor[:,-remain:])

    return sample_list


def split_sample(tensor, window=5, step=1, offset=0, keep_tail=True, merge_remain=False):
    """
    :param tensor: 2d numpy or 2d tensor
    :param window: int, size of window default=5
    :param step: int, size between two windows default=1
    :param offset: int, first window offset default=0
    :param keep_tail: Boolean , {True : save tail of data,; False : possible not save tail of data} default True
    :param merge_remain: Boolean , {True: merge the ending group with its previous group, if the ending group size is less than 'window'
                                    False: do not merge, but discard the ending group, if the ending group size is less than 'window'}
                        (keep_tail determines whether the smaller sized group, if any, is at the front or the end)
    :return: list within numpy data
    Examples::
        (note: the example data is not in the correct input format; it's intended for understanding only)
        >>> # use to split data set
        >>> import numpy as np
        >>> data = np.array(range(1, 11))
        >>> window_train = 5
        >>> window_test = 3
        >>> # keep_tail=False, merge_remain=False 
        >>> train_data = split_sample(data, window=window_train, step=window_test, offset=0, keep_tail=False, merge_remain=False)
        >>> train_data
        [array([1, 2, 3, 4, 5]), array([4, 5, 6, 7, 8])]
        >>> test_data = split_sample(data, window=window_test, step=window_test, offset=window_train, keep_tail=False, merge_remain=True)
        [array([ 6,  7,  8,  9, 10])]
        
        >>> # use to split sample
        >>> data = np.array(range(30)).reshape(6, 5)
        >>> # keep_tail=True, merge_remain=False
        >>> sample1 = split_sample(data, window=3, step=2, offset=0, keep_tail=True, merge_remain=False)
        >>> sample1
        [array([[ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]]),
         array([[15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29]])]
        
        >>> # keep_tail=False, merge_remain=False
        >>> sample2 = split_sample(data, window=3, step=2, offset=0, keep_tail=False, merge_remain=False)
        >>> sample2
        [array([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14]]),
         array([[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])]
        >>> # keep_tail=True, merge_remain=False
        >>> sample2 = split_sample(data, window=3, step=2, offset=0, keep_tail=False, merge_remain=True)
        >>> sample2
        [array([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14]]),
         array([[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])]
    """
    index = int((len(tensor) - window - offset) / step) + 1
    remain = len(tensor) - window - offset - (index - 1) * step
    sample_list = split(tensor, window=window, step=step, offset=offset, keep_tail=keep_tail)
    if remain:
        if keep_tail:
            idx = 1
        else:
            idx = -1
        
        if not merge_remain:
            return sample_list[idx:] if idx==1 else sample_list[:idx]
        else:
            
            if isinstance(tensor, torch.Tensor):
                cat_func = torch.cat
            else:
                cat_func = np.concatenate
            
            sample_list[idx-1] = cat_func([sample_list[idx-1], sample_list[idx]])
            del sample_list[idx]
            return sample_list 
    else:
        return sample_list 


def split_sample3d(tensor, window=5, step=1, offset=0, keep_tail=True, merge_remain=False, dim=1):
    """
    :param tensor: 3d numpy or 3d tensor
    :param window: int, size of window default=5
    :param step: int, size between two windows default=1
    :param offset: int, first window offset default=0
    :param keep_tail: Boolean , {True : save tail of data,; False : possible not save tail of data} default True
    :param merge_remain: Boolean , {True: and if keep_tail is True, the first sample include remain sample, 
                                           elif keep_tail is Flase, the last sample include remain sample.
                                 Flase: the sample decide by value of keep_tail
                                }
    :return: list within numpy data
    Examples::
        (note: the example data is not in the correct input format; it's intended for understanding only)
        >>> # use to split data set
        >>> import numpy as np
        >>> data = np.array(range(1, 11))
        >>> window_train = 5
        >>> window_test = 3
        >>> # keep_tail=False, merge_remain=False 
        >>> train_data = split_sample(data, window=window_train, step=window_test, offset=0, keep_tail=False, merge_remain=False)
        >>> train_data
        [array([1, 2, 3, 4, 5]), array([4, 5, 6, 7, 8])]
        >>> test_data = split_sample(data, window=window_test, step=window_test, offset=window_train, keep_tail=False, merge_remain=True)
        [array([ 6,  7,  8,  9, 10])]
        
        >>> # use to split sample
        >>> data = np.array(range(30)).reshape(6, 5)
        >>> # keep_tail=True, merge_remain=False
        >>> sample1 = split_sample(data, window=3, step=2, offset=0, keep_tail=True, merge_remain=False)
        >>> sample1
        [array([[ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]]),
         array([[15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29]])]
        
        >>> # keep_tail=False, merge_remain=False
        >>> sample2 = split_sample(data, window=3, step=2, offset=0, keep_tail=False, merge_remain=False)
        >>> sample2
        [array([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14]]),
         array([[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])]
    """
    length = tensor.shape[dim]
    index = int((length - window - offset) / step) + 1
    remain = length - window - offset - (index - 1) * step
    sample_list = split3d(tensor, window=window, step=step, offset=offset, keep_tail=keep_tail)
    if remain:
        if keep_tail:
            idx = 1
        else:
            idx = -1
        
        if not merge_remain:
            return sample_list[idx:] if idx==1 else sample_list[:idx]
        else:
            #pdb.set_trace()
            if isinstance(tensor, torch.Tensor):
                cat_func = torch.cat
            else:
                cat_func = np.concatenate
                
            sample_list[idx-1] = cat_func([sample_list[idx-1], sample_list[idx]],dim)
            del sample_list[idx]
            return sample_list 
    else:
        return sample_list 


if __name__ == '__main__':
    np.random.seed(520)
    data2d_np = np.random.randn(10,3)
    data2d_ts = torch.tensor(data2d_np, dtype=torch.float32)
    
    data_list_by_split_np = split(data2d_np)
    data_list_by_split_ts = split(data2d_ts)
    
    data3d_np = np.random.randint(1,5,(2,10,3))
    data3d_ts = torch.tensor(data3d_np, dtype=torch.int32)
    data_list_by_split3d_np = split3d(data3d_np)
    data_list_by_split3d_ts = split3d(data3d_ts)
    
    #
    data_sample_list_np = split_sample(data2d_np, window=3)
    data_sample_list_ts = split_sample(data2d_ts, window=3)
    
    data_sample3d_list_np = split_sample3d(data3d_np, window=3, step=2, merge_remain=False)
    data_sample3d_list_ts = split_sample3d(data3d_ts, window=3, step=2, merge_remain=False)
    
    data_sample3d_list_np2 = split_sample3d(data3d_np, window=3, step=2, merge_remain=True)
    data_sample3d_list_ts2 = split_sample3d(data3d_ts, window=3, step=2, merge_remain=True)
