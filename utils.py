import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
# from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st

import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
import numpy as np
import matplotlib.colors as colors
import os
import pdb

import constants


def get_anomaly_score(detection_model, data, device):
    with torch.no_grad():
        test_x = torch.Tensor(data)
        test_x = test_x.to(device)

        z, log_jac_det = detection_model(test_x)
        # loss = torch.sum(z**2, 1) - log_jac_det
        # loss = loss.mean() / (constants.N_DIM*constants.SEQ_DIM*(1+constants.L*2))
        # loss = torch.sum(z**2, 1)
        
        train_mean = torch.zeros(constants.N_DIM*constants.SEQ_DIM*(1+constants.L*2)).to(device)
        train_cov = torch.eye(constants.N_DIM*constants.SEQ_DIM*(1+constants.L*2)).to(device)
        base_dist = MultivariateNormal(train_mean, train_cov)
        loss = -base_dist.log_prob(z) - log_jac_det
        
        
        loss = loss / (constants.N_DIM*constants.SEQ_DIM*(1+constants.L*2))
        anomaly_scores = loss.cpu().numpy()

        # auc = roc_auc_score(test_labels, anomaly_scores)
        # print(f"AUROC: {auc}")
    return anomaly_scores


class DataProcessor():
    def __init__(self) -> None:
        pass
    
    def two_pi_warp(self, angles):
        twp_pi = 2 * np.pi
        return (angles + twp_pi) % (twp_pi)
        # twp_pi = 2 * np.pi
        # if angle > twp_pi:
        #     return angle - twp_pi
        # elif angle < 0:
        #     return angle + twp_pi
        # else:
        #     return angle
    
    def data_normalize(self, data):
        data = data.flatten()
        data_min = np.min(data)

        # data_min = 9. # TODO

        data = data - data_min
        data_max = np.max(data)

        # data_max = 2. # TODO

        # data = data / data_max
        # data = data.reshape(-1, constants.SEQ_DIM)

        return [data_max, data_min]
    
    def runtime_normalize(self, data, params):
        data = data.flatten()
        result = (data - params[1]) / params[0]
        result = result.reshape(-1, constants.SEQ_DIM)
        return result
    
    def de_normalize(self, data, params):
        return data * params[0] + params[1]
    

class torchDataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        #TODO: array input sequence
        self.data_array = torch.from_numpy(data).type('torch.FloatTensor')
        if constants.MOVE_DATA_TO_DEVICE: self.data_array = self.data_array.to(device)
        print('data_array.shape', self.data_array.shape)

    def __len__(self):
        return self.data_array.shape[0]
    
    def __getitem__(self, index):
        return self.data_array[index]


def data_generate(mu):
    device = torch.device(constants.CUDA_NAME)
    dt = 0.005
    t = torch.arange(0., 5., dt).to(device)

    u = np.array([1., 1.])
    p = parameters_vehicle1()
    true_y = np.zeros((t.shape[0], 1, 7))
    true_y[0, 0, :] = np.array([0, 0, 0, 15, 0, 0, 0])
    for i in range(1, len(t)):
        true_y[i, 0, :] = true_y[i-1, 0, :] + np.array(vehicle_dynamics_st(true_y[i-1, 0, :], u, p, mu)) * dt
    true_y = torch.tensor(true_y).float().to(device)

    return true_y, t
    
    
def get_data_inference(input_data, normalization_intervals):
    np.random.seed(42)
    device = torch.device(constants.CUDA_NAME)
    data_processor = DataProcessor()

    positional_encoding = PositionalEncoding(L = constants.L)

    input_data = input_data[constants.DROP_FIRST:]

    # input_data = np.vstack([input_data[i:i+constants.SEQ_DIM].flatten() for i in range(0, len(input_data)-constants.SEQ_DIM+1, constants.SEQ_DIM)])
    # input_data = np.vstack([input_data[i:i+constants.SEQ_DIM].flatten() for i in range(len(input_data)-constants.SEQ_DIM+1)])

    input_data = np.vstack([input_data[i:i+constants.SEQ_DIM][None, :] for i in range(0, len(input_data)-constants.SEQ_DIM+1, constants.SEQ_DIM)])
    train_data_seq = input_data

    for i in range(constants.N_DIM):
        train_data_seq[:, :, i] = data_processor.runtime_normalize(train_data_seq[:, :, i], normalization_intervals[i])

    train_data_seq = train_data_seq.reshape(train_data_seq.shape[0], constants.N_DIM*constants.SEQ_DIM)

    sin_part, cos_part = positional_encoding.encode(train_data_seq[:,:,None]) # shape: [steps, N_dim, L]
    sin_part, cos_part = sin_part.reshape(train_data_seq.shape[0], constants.N_DIM*constants.L*constants.SEQ_DIM), cos_part.reshape(train_data_seq.shape[0], constants.N_DIM*constants.L*constants.SEQ_DIM)

    train_data_seq = np.hstack((train_data_seq, sin_part))
    train_data_seq = np.hstack((train_data_seq, cos_part))


    return train_data_seq


def get_data(input_data, ad_score=False, train_ratio=0.8, normalization_intervals=[]):
    np.random.seed(42)
    device = torch.device(constants.CUDA_NAME)
    data_processor = DataProcessor()

    positional_encoding = PositionalEncoding(L = constants.L)

    input_data = input_data[constants.DROP_FIRST:]

    # input_data = np.vstack([input_data[i:i+constants.SEQ_DIM].flatten() for i in range(0, len(input_data)-constants.SEQ_DIM+1, constants.SEQ_DIM)])
    # input_data = np.vstack([input_data[i:i+constants.SEQ_DIM].flatten() for i in range(len(input_data)-constants.SEQ_DIM+1)])

    input_data = np.vstack([input_data[i:i+constants.SEQ_DIM][None, :] for i in range(0, len(input_data)-constants.SEQ_DIM+1, constants.SEQ_DIM)])

    time_steps = input_data.shape[0]

    num_selected = int(train_ratio * time_steps)
    selected_rows = np.random.choice(time_steps, size=num_selected, replace=False)
    remaining_rows = np.setdiff1d(np.arange(time_steps), selected_rows)

    train_data_seq = input_data[selected_rows]
    test_data_seq = input_data[remaining_rows]

    if len(normalization_intervals) == 0:
        normalization_intervals = np.empty((constants.N_DIM, 2))
        for i in range(constants.N_DIM):
            [normalization_intervals[i, 0], normalization_intervals[i, 1]] = data_processor.data_normalize(train_data_seq[:, :, i])
            # if i == 0:
            #     [normalization_intervals[i, 0], normalization_intervals[i, 1]] = [5., 7.5] # NOTE
            train_data_seq[:, :, i] = data_processor.runtime_normalize(train_data_seq[:, :, i], normalization_intervals[i])
    else:
        for i in range(constants.N_DIM):
            train_data_seq[:, :, i] = data_processor.runtime_normalize(train_data_seq[:, :, i], normalization_intervals[i])

    for i in range(constants.N_DIM):
        test_data_seq[:, :, i] = data_processor.runtime_normalize(test_data_seq[:, :, i], normalization_intervals[i])


    train_data_seq = train_data_seq.reshape(train_data_seq.shape[0], constants.N_DIM*constants.SEQ_DIM)
    test_data_seq = test_data_seq.reshape(test_data_seq.shape[0], constants.N_DIM*constants.SEQ_DIM)

    sin_part, cos_part = positional_encoding.encode(train_data_seq[:,:,None]) # shape: [steps, N_dim, L]
    sin_part, cos_part = sin_part.reshape(train_data_seq.shape[0], constants.N_DIM*constants.L*constants.SEQ_DIM), cos_part.reshape(train_data_seq.shape[0], constants.N_DIM*constants.L*constants.SEQ_DIM)

    train_data_seq = np.hstack((train_data_seq, sin_part))
    train_data_seq = np.hstack((train_data_seq, cos_part))

    sin_part, cos_part = positional_encoding.encode(test_data_seq[:,:,None]) # shape: [steps, N_dim, L]
    sin_part, cos_part = sin_part.reshape(test_data_seq.shape[0], constants.N_DIM*constants.L*constants.SEQ_DIM), cos_part.reshape(test_data_seq.shape[0], constants.N_DIM*constants.L*constants.SEQ_DIM)
    test_data_seq = np.hstack((test_data_seq, sin_part))
    test_data_seq = np.hstack((test_data_seq, cos_part))


    if ad_score:
        return train_data_seq, test_data_seq

    train_set = torchDataset(train_data_seq, device)
    test_set = torchDataset(test_data_seq, device)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=constants.BATCHSIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=constants.BATCHSIZE, shuffle=True, drop_last=False)

    return train_loader, test_loader, normalization_intervals


def filter_data(data, controls_data=None):
    # TODO mask1 = data[:, 4] > 0.8len(data) # 0.6 fric has one abnormal spike

    mask1 = data[:, 4] > 3
    # mask1 = data[:, 4] > 0.3
    mask2 = data[:, 4] < -3
    mask3 = data[:, 0] > 9
    mask_ = mask1 | mask2 | mask3
    
    control_length = 20
    mask_ = np.floor_divide(np.where(mask_)[0], control_length) * control_length
    mask_ = np.unique(mask_)

    filtered_array = [i for value in mask_ for i in range(value, value + 20)]
    filtered_array = np.array(filtered_array)
    print('filtered numpy: ', filtered_array.shape)

    if controls_data is None:
        if len(filtered_array) > 0:
            filtered_data = np.delete(data, filtered_array, axis=0)
        else:
            filtered_data = data
        return filtered_data
    else:
        if len(filtered_array) > 0:
            filtered_data = np.delete(data, filtered_array, axis=0)
            filtered_controls_data = np.delete(controls_data, filtered_array, axis=0)
        else:
            filtered_data = data
            filtered_controls_data = controls_data
        return filtered_data, filtered_controls_data

class PositionalEncoding():
    def __init__(self, L):
        self.L = L
        self.val_list = []
        for l in range(L):
            self.val_list.append(2.0 ** l)
        self.val_list = np.array(self.val_list)

    def encode(self, x):
        return np.sin(self.val_list * np.pi * x), np.cos(self.val_list * np.pi * x)
    
    def encode_even(self, x):
        return np.sin(self.val_list * np.pi * 2 * x), np.cos(self.val_list * np.pi * 2 * x)
    
    def decode(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / (np.pi)
        if np.isscalar(atan2_value) == 1:
            if atan2_value > 0:
                return atan2_value
            else:
                return 1 + atan2_value
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            return atan2_value
        
    def decode_even(self, sin_value, cos_value):
        atan2_value = np.arctan2(sin_value, cos_value) / np.pi/2
        if np.isscalar(atan2_value) == 1:
            if atan2_value < 0:
                atan2_value = 1 + atan2_value
            if np.abs(atan2_value - 1) < 0.001:
                atan2_value = 0
        else:
            atan2_value[np.where(atan2_value < 0)] = atan2_value[np.where(atan2_value < 0)] + 1
            atan2_value[np.where(np.abs(atan2_value - 1) < 0.001)] = 0
        return atan2_value