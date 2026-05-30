#_%% [markdown]
# # DeepONet Test 

#_%%
import os
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset as TorchDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import pearsonr
from math import sqrt
import scipy.fft as ft
import random
from torch.cuda.amp import GradScaler, autocast
import h5py
import gc
import torch.multiprocessing as mp
import shutil
import Module_DL_Helper as MDLH

#_%%
# store starting time
overall_starting_time = time.time()

#_%%
# define enum for input options
class InputOptions(Enum):
    Load = 1
    Strain = 2
    Load_Strain = 3

#_%%
load_folder_path   = f'../2-Processed_Generated_Data'
output_folder_path = f'../2-Processed_Generated_Data'
strain_folder_path = f'../2-Processed_Generated_Data'

load_file_name = f'Grid_3D_BodySource_V004_force_per_history_increment_grid_LC-'
output_file_name = f'all_data_array_grid_LC-'
strain_file_name = f'strain-center_data_array_grid_LC-'

coordinates_file_name = f'coordinates_array_grid_LC-0-100'
normalization_option = 2 # 0: no normalization, 1: normalize between 0 and 1, 2: normalize between -1 and 1 3: standardization 4: divide by max value 5: divide by std
loss_criterion_option = 7 # 1 : MSELoss('mean') 2 : L1Loss('mean') 3 : SmoothL1Loss('mean') 4 : MSELoss('sum') 5 : L1Loss('sum') 6 : SmoothL1Loss('sum') 7 : CustomLoss (L2) 8 : CustomLoss (Normalized L2)
trial_save_number = 'T001'
LC_start = 0
LC_end = 900
max_items_saved = 900
dl_batch_size = 16
selected_input_option = InputOptions.Load_Strain
split_in_branch = True
enforce_BC = True
train_for_theta_only = True
read_all_the_file_once = True # (True Preferred for HPC) works only if max_items_saved is > (LC_end - LC_start)
use_torch_multiprocessing_manager = False
epochs_sessions_list = [10,30,60,4000-100,4000,4000,4000,4000,4000]
device_save_name = 'hpc'
slicing_settings = {'load_time_start': 1,
                    'load_time_skip': 1,
                    'load_time_end': None,
                    'strain_time_start': 0,
                    'strain_time_skip': 1,
                    'strain_time_end': -1,
                    'point_load_start': (1,1,1),
                    'point_load_skip': (4,4,4),
                    'point_load_end': (-1,-1,-1),
                    'point_load_flatten': True,
                    'point_strain_start': (0,0,0),
                    'point_strain_skip': (4,4,4),
                    'point_strain_end': (-1,-1,-1),
                    'point_strain_flatten': True,
                    'point_output_start': (0,0,0), #(0,0,0),
                    'point_output_skip': (3,3,3),
                    'point_output_end': (None,None,None), #(None,None,None),
                    'point_output_flatten': True}
extend_on_previous_model = False # if True, it will load the previous model and continue training
logs_save_folder = f'logs'
models_save_folder = f'models'
sco_save_folder = f'sco_files'
parent_plots_save_folder = f'plots'
previous_model_path = f'./{models_save_folder}/model_007_V002_A005_theta_only_load_strain_dev1_LC_0-400_last_train.pth' # if extend_on_previous_model is True, it will load the model from this path


is_read_only = False
trial_save_text = f'{trial_save_number}_{device_save_name}_LC_{LC_start}-{LC_end}'
last_model_path=f'./{models_save_folder}/model_007_{trial_save_text}_last_train.pth'
best_model_path_train=f'./{models_save_folder}/model_007_{trial_save_text}_best_train.pth'
best_model_path_val = f'./{models_save_folder}/model_007_{trial_save_text}_best_val.pth'

if is_read_only:
    # read only settings (for plot or further results analysis)
    epochs_sessions_list = []
    read_save_text = f'{trial_save_number}(R)_{device_save_name}_LC_{LC_start}-{LC_end}'
    plots_save_folder = f'{parent_plots_save_folder}/model_007_{read_save_text}'
    sco_file_path = f'./{sco_save_folder}/model_007_{read_save_text}_sco.py'
    log_file_name = f'model_007_{read_save_text}_log.log'
else:    
    # new model
    plots_save_folder = f'{parent_plots_save_folder}/model_007_{trial_save_text}'
    sco_file_path = f'./{sco_save_folder}/model_007_{trial_save_text}_sco.py'
    log_file_name = f'model_007_{trial_save_text}_log.log'

#_%%
############################
## input for testing data
############################
plot_results_for_testing = True
tst_LC_start = 900
tst_LC_end = 1000
tst_max_items_saved = 200
tst_batch_size = 20
# clone the slicing settings
tst_slicing_settings = slicing_settings.copy()
tst_slicing_settings['point_output_start'] = [0,0,0]
tst_slicing_settings['point_output_skip'] = [1,1,1]
tst_slicing_settings['point_output_end'] = [None,None,None]

# Check if the folder exists, and create it if it doesn't
for folder_name in [logs_save_folder, models_save_folder, plots_save_folder,sco_save_folder]:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
# Create logger object
my_logger = MDLH.MyLogger(f'./{logs_save_folder}/{log_file_name}', max_wait_list_length = 10, use_csv=False)

# Copy the current script to the sco folder
current_file_path = os.path.abspath(__file__)
shutil.copy(current_file_path, sco_file_path)

# Create manager object
if use_torch_multiprocessing_manager:
    pytorch_manager_1 = mp.Manager()
    main_cashed_items_dict = pytorch_manager_1.dict()
    pytorch_manager_2 = mp.Manager()
    test_cashed_items_dict = pytorch_manager_2.dict()
else:
    main_cashed_items_dict = {}
    test_cashed_items_dict = {}


#_%%
# Set as global variable in the helper module
MDLH.main_cashed_items_dict = main_cashed_items_dict
MDLH.test_cashed_items_dict = test_cashed_items_dict
MDLH.use_torch_multiprocessing_manager = use_torch_multiprocessing_manager
MDLH.my_logger = my_logger
MDLH.plots_save_folder = plots_save_folder
MDLH.models_save_folder = models_save_folder
MDLH.last_model_path = last_model_path
MDLH.train_for_theta_only = train_for_theta_only

#_%%
data_loading_settings = {
    "load_folder_path": load_folder_path,
    "output_folder_path": output_folder_path,
    "strain_folder_path": strain_folder_path,
    "load_file_name": load_file_name,
    "output_file_name": output_file_name,
    "strain_file_name": strain_file_name,
    "coordinates_file_name": coordinates_file_name,
    "LC_start": LC_start,
    "LC_end": LC_end,
    "slicing_settings": slicing_settings,
    "normalization_option": normalization_option,
    "max_items_saved": max_items_saved,
    "read_all_the_file_once": read_all_the_file_once
}

#_%%
# load the dataset
my_dataset = MDLH.MyDataset(data_loading_settings, mode = 'train')
# split the dataset 
my_dataset_len = len(my_dataset)
train_dataset_len = int(0.8 * my_dataset_len)
val_dataset_len = my_dataset_len - train_dataset_len
#train_dataset, val_dataset = random_split(my_dataset, [train_dataset_len,val_dataset_len], generator=torch.Generator().manual_seed(42))
train_dataset = torch.utils.data.Subset(my_dataset, range(train_dataset_len))
val_dataset = torch.utils.data.Subset(my_dataset, range(train_dataset_len, my_dataset_len))
train_dataset = torch.utils.data.Subset(my_dataset, range(train_dataset_len))
val_dataset = torch.utils.data.Subset(my_dataset, range(train_dataset_len, my_dataset_len))
my_logger.add_to_wait_list(f'length of my_dataset: {len(my_dataset)}')
my_logger.add_to_wait_list(f'length of train_dataset: {len(train_dataset)}')
my_logger.add_to_wait_list(f'length of val_dataset: {len(val_dataset)}')

#_%%
# print shapes of the datasets at selected indices
for i in [my_dataset_len-1,int(my_dataset_len*0.5),0]:
    load_data_i, strain_data_i, output_data_i = my_dataset[i]
    my_logger.add_to_wait_list(f'load_data_{i} shape: {load_data_i.shape}')
    my_logger.add_to_wait_list(f'strain_data_{i} shape: {strain_data_i.shape}')
    my_logger.add_to_wait_list(f'output_data_{i} shape: {output_data_i.shape}')
    my_logger.add_to_wait_list(f'load_data_{i}=> max: {torch.max(load_data_i)}, min: {torch.min(load_data_i)}, mean: {torch.mean(load_data_i)}, std: {torch.std(load_data_i)}')
    my_logger.add_to_wait_list(f'strain_data_{i}=> max: {torch.max(strain_data_i)}, min: {torch.min(strain_data_i)}, mean: {torch.mean(strain_data_i)}, std: {torch.std(strain_data_i)}')
    my_logger.add_to_wait_list(f'output_data_{i}[0]=> max: {torch.max(output_data_i[...,0])} , min: {torch.min(output_data_i[...,0])}, mean: {torch.mean(output_data_i[...,0])}, std: {torch.std(output_data_i[...,0])}')
    if train_for_theta_only == False:
        my_logger.add_to_wait_list(f'output_data_{i}[1]=> max: {torch.max(output_data_i[...,1])} , min: {torch.min(output_data_i[...,1])}, mean: {torch.mean(output_data_i[...,1])}, std: {torch.std(output_data_i[...,1])}')
        my_logger.add_to_wait_list(f'output_data_{i}[2]=> max: {torch.max(output_data_i[...,2])} , min: {torch.min(output_data_i[...,2])}, mean: {torch.mean(output_data_i[...,2])}, std: {torch.std(output_data_i[...,2])}')
        my_logger.add_to_wait_list(f'output_data_{i}[3]=> max: {torch.max(output_data_i[...,3])} , min: {torch.min(output_data_i[...,3])}, mean: {torch.mean(output_data_i[...,3])}, std: {torch.std(output_data_i[...,3])}')
    load_data_i_shape = load_data_i.shape
    strain_data_i_shape = strain_data_i.shape
    output_data_i_shape = output_data_i.shape

del load_data_i
del strain_data_i
del output_data_i

# print original shapes
my_logger.add_to_wait_list(f'original_load_grid_shape: {my_dataset.original_load_grid_shape}')
my_logger.add_to_wait_list(f'original_strain_grid_shape: {my_dataset.original_strain_grid_shape}')
my_logger.add_to_wait_list(f'original_output_grid_shape: {my_dataset.original_output_grid_shape}')

#_%%
# set trunk input for all batches
global_trunk_input_tensor = my_dataset.get_coordinates_array(return_as_tensor=True)
my_logger.add_to_wait_list(f'shape of the points_coordinates_tensor: {global_trunk_input_tensor.shape}')

#_%%
# define the dataloaders
dl_use_persistent_workers = False
dl_num_workers = 0 # number of processes to use for data loading
dl_prefetch_factor = 2 # number of samples loaded in advance by each worker
# NOTE: data loaders are re-defined after training with [{(shuffle = False)}] to get the same order of the data for testing
train_dataloader    = DataLoader(train_dataset, batch_size=dl_batch_size, shuffle=True , num_workers=dl_num_workers, prefetch_factor=dl_prefetch_factor, persistent_workers=dl_use_persistent_workers)
train_dataloader_NS = DataLoader(train_dataset, batch_size=dl_batch_size, shuffle=False, num_workers=dl_num_workers, prefetch_factor=dl_prefetch_factor, persistent_workers=dl_use_persistent_workers)
val_dataloader      = DataLoader(val_dataset  , batch_size=dl_batch_size, shuffle=False, num_workers=dl_num_workers, prefetch_factor=dl_prefetch_factor, persistent_workers=dl_use_persistent_workers)
# train_dataloader_NS For prediction plots (consistent sorted data is needed)
#_%%
train_batch_input_load, train_batch_input_strain, train_output_target = next(iter(train_dataloader_NS))
my_logger.add_to_wait_list(f'train_batch_input_load: {train_batch_input_load.shape}')
my_logger.add_to_wait_list(f'train_batch_input_strain: {train_batch_input_strain.shape}')
my_logger.add_to_wait_list(f'train_output_target: {train_output_target.shape}')

#_%%
# inspect some nodes
dim_x = 1.0
dim_y = 1.0
dim_z = 1.0

target_coord_list = []
target_coord_list.append(np.array([dim_x*0.5, dim_y*0.5, dim_z*0.5]))
target_coord_list.append(np.array([dim_x*1.0, dim_y*0.0, dim_z*0.5]))
target_coord_list.append(np.array([dim_x*1.0, dim_y*1.0, dim_z*0.5]))
target_coord_list.append(np.array([dim_x*0.5, dim_y*1.0, dim_z*0.5]))
target_coord_list.append(np.array([dim_x*0.0, dim_y*0.5, dim_z*0.5]))          

computed_index_list = my_dataset.get_index_of_coordinates_list(target_coord_list)
computed_dim_list = global_trunk_input_tensor[computed_index_list, :]

for i in range(len(target_coord_list)):
    my_logger.add_to_wait_list(f'computed_index_list[{i}]: {computed_index_list[i]} ==> target_coord_list[{i}]: {target_coord_list[i]} ==> computed_dim_list[{i}]: {computed_dim_list[i]}')

#_%%
my_logger.log_the_wait_list()

#_%%
# specify nodes of interest
node_indices = computed_index_list

#######################################
## starting testing implementation
#######################################
if plot_results_for_testing:
    # clone the data_loading_settings
    tst_data_loading_settings = data_loading_settings.copy()
    tst_data_loading_settings['LC_start'] = tst_LC_start
    tst_data_loading_settings['LC_end'] = tst_LC_end
    tst_data_loading_settings['max_items_saved'] = tst_max_items_saved
    tst_data_loading_settings['slicing_settings'] = tst_slicing_settings
    
    tst_dataset = MDLH.MyDataset(tst_data_loading_settings, mode = 'test')
    tst_dataset_len = len(tst_dataset)
    my_logger.add_to_wait_list(f'Testing: length of tst_dataset: {tst_dataset_len}')
    tst_dataloader = DataLoader(tst_dataset, batch_size=tst_batch_size, shuffle=False, num_workers=dl_num_workers, prefetch_factor=dl_prefetch_factor, persistent_workers=dl_use_persistent_workers)
    tst_trunk_input_tensor = tst_dataset.get_coordinates_array(return_as_tensor=True)
    my_logger.add_to_wait_list(f'Testing: shape of the points_coordinates_tensor: {tst_trunk_input_tensor.shape}')
    
    # print shapes of the datasets at selected indices
    for i in [tst_dataset_len-1,int(tst_dataset_len*0.5),0]:
        load_data_i, strain_data_i, output_data_i = tst_dataset[i]
        my_logger.add_to_wait_list(f'Testing: load_data_{i} shape: {load_data_i.shape}')
        my_logger.add_to_wait_list(f'Testing: strain_data_{i} shape: {strain_data_i.shape}')
        my_logger.add_to_wait_list(f'Testing: output_data_{i} shape: {output_data_i.shape}')
        my_logger.add_to_wait_list(f'Testing: load_data_{i}=> max: {torch.max(load_data_i)}, min: {torch.min(load_data_i)}, mean: {torch.mean(load_data_i)}, std: {torch.std(load_data_i)}')
        my_logger.add_to_wait_list(f'Testing: strain_data_{i}=> max: {torch.max(strain_data_i)}, min: {torch.min(strain_data_i)}, mean: {torch.mean(strain_data_i)}, std: {torch.std(strain_data_i)}')
        my_logger.add_to_wait_list(f'Testing: output_data_{i}[0]=> max: {torch.max(output_data_i[...,0])} , min: {torch.min(output_data_i[...,0])}, mean: {torch.mean(output_data_i[...,0])}, std: {torch.std(output_data_i[...,0])}')
        if train_for_theta_only == False:
            my_logger.add_to_wait_list(f'Testing: output_data_{i}[1]=> max: {torch.max(output_data_i[...,1])} , min: {torch.min(output_data_i[...,1])}, mean: {torch.mean(output_data_i[...,1])}, std: {torch.std(output_data_i[...,1])}')
            my_logger.add_to_wait_list(f'Testing: output_data_{i}[2]=> max: {torch.max(output_data_i[...,2])} , min: {torch.min(output_data_i[...,2])}, mean: {torch.mean(output_data_i[...,2])}, std: {torch.std(output_data_i[...,2])}')
            my_logger.add_to_wait_list(f'Testing: output_data_{i}[3]=> max: {torch.max(output_data_i[...,3])} , min: {torch.min(output_data_i[...,3])}, mean: {torch.mean(output_data_i[...,3])}, std: {torch.std(output_data_i[...,3])}')
        tst_load_data_i_shape = load_data_i.shape
        tst_strain_data_i_shape = strain_data_i.shape
        tst_output_data_i_shape = output_data_i.shape

    del load_data_i
    del strain_data_i
    del output_data_i
    
    # print original shapes
    my_logger.add_to_wait_list(f'original_load_grid_shape: {tst_dataset.original_load_grid_shape}')
    my_logger.add_to_wait_list(f'original_strain_grid_shape: {tst_dataset.original_strain_grid_shape}')
    my_logger.add_to_wait_list(f'original_output_grid_shape: {tst_dataset.original_output_grid_shape}')
    
    tst_computed_index_list = tst_dataset.get_index_of_coordinates_list(target_coord_list)
    tst_computed_dim_list = tst_trunk_input_tensor[tst_computed_index_list, :]
    for i in range(len(target_coord_list)):
        my_logger.add_to_wait_list(f'Testing: computed_index_list[{i}]: {tst_computed_index_list[i]} ==> target_coord_list[{i}]: {target_coord_list[i]} ==> computed_dim_list[{i}]: {tst_computed_dim_list[i]}')

    tst_node_indices = tst_computed_index_list
    
else:
    tst_dataset = None
    tst_dataloader = None
    tst_trunk_input_tensor = None
    tst_node_indices = None
    my_logger.add_to_wait_list(f'Testing: Is not requested')
    
my_logger.log_the_wait_list()

#_%%
######################################################################################
######################################################################################
# MODEL DESIGN
######################################################################################
######################################################################################
general_dropout = 0.0
use_scheduler = True

# switch based on selected_input_option
if selected_input_option == InputOptions.Load:
    branch_input_dim_a = load_data_i_shape[-1]
    branch_input_dim_b = 0
elif selected_input_option == InputOptions.Strain:
    branch_input_dim_a = 0
    branch_input_dim_b = strain_data_i_shape[-1]
elif selected_input_option == InputOptions.Load_Strain:
    branch_input_dim_a = load_data_i_shape[-1]
    branch_input_dim_b = strain_data_i_shape[-1]

n_channels_a = 0
n_channels_b = 25
n_groups_a = 1
n_groups_b = 25

hidden_temporal_size_a = 200
hidden_temporal_size_b = 50
temporal_num_layers_a = 2
temporal_num_layers_b = 2


trunk_input_dim = 3  # coordinate_x, coordinate_y, coordinate_z
trunk_internal_dim = 200

branch_output_dim_a = 200
branch_output_dim_b = 50
trunk_output_dim = branch_output_dim_a + branch_output_dim_b

# one of branch_n_components and trunk_n_components should be 1
if split_in_branch:
    branch_n_components = output_data_i_shape[-1]
    trunk_n_components = 1 
else:   
    branch_n_components = 1
    trunk_n_components = output_data_i_shape[-1]


branch_input_dim = [branch_input_dim_a, branch_input_dim_b]
hidden_temporal_size = [hidden_temporal_size_a, hidden_temporal_size_b]
temporal_num_layers = [temporal_num_layers_a, temporal_num_layers_b]
branch_output_dim = [branch_output_dim_a, branch_output_dim_b]

#_%%
# DeepONet Branch Net
class BranchNet_A(nn.Module):
    def __init__(self, input_dim, hidden_temporal_size_a, temporal_num_layers_a, output_dim, n_components):
        super(BranchNet_A, self).__init__()
        self.n_components = n_components
        self.output_dim = output_dim
        self.temporal_layer = nn.GRU(input_dim, hidden_temporal_size_a, num_layers=temporal_num_layers_a, bias=True, batch_first=True, dropout=0.0, bidirectional=False)
        self.n_channels = n_channels_a
        self.n_groups = n_groups_a
        if self.n_channels > 0:
            assert hidden_temporal_size_a % self.n_channels == 0
            self.features_per_channel = hidden_temporal_size_a // self.n_channels
            self.group_norm = nn.GroupNorm(self.n_groups, self.n_channels) 
        self.dropout1 = nn.Dropout(p=general_dropout)  # Dropout layer
        self.fc1 = nn.Linear(hidden_temporal_size_a, output_dim  * n_components)

    def forward(self, x):
        x,_ = self.temporal_layer(x)  
        if self.n_channels > 0:
            x_org_shape = x.shape      
            x = x.reshape(-1, self.n_channels, self.features_per_channel)
            x = self.group_norm(x)
            x = x.reshape(x_org_shape)
        x = self.dropout1(x)
        x = self.fc1(x)      # (batch_size , sequence length , output_dim * n_components)
        if split_in_branch:
            x = x.view(*x.shape[:-1], self.output_dim, self.n_components)  # (batch_size , sequence length , output_dim , n_components)
        return x
    
class BranchNet_B(nn.Module):
    def __init__(self, input_dim, hidden_temporal_size_b, temporal_num_layers_b, output_dim, n_components):
        super(BranchNet_B, self).__init__()
        self.n_components = n_components
        self.output_dim = output_dim
        self.temporal_layer = nn.GRU(input_dim, hidden_temporal_size_b, num_layers=temporal_num_layers_b, bias=True, batch_first=True, dropout=0.0, bidirectional=False)
        self.n_channels = n_channels_b
        self.n_groups = n_groups_b
        if self.n_channels > 0:
            assert hidden_temporal_size_b % self.n_channels == 0
            self.features_per_channel = hidden_temporal_size_b // self.n_channels
            self.grou_norm = nn.GroupNorm(self.n_groups, self.n_channels)
        self.dropout1 = nn.Dropout(p=general_dropout)  # Dropout layer
        self.fc1 = nn.Linear(hidden_temporal_size_b, output_dim  * n_components)

    def forward(self, x):
        x,_ = self.temporal_layer(x)
        if self.n_channels > 0:
            x_org_shape = x.shape      
            x = x.reshape(-1, self.n_channels, self.features_per_channel)
            x = self.grou_norm(x)
            x = x.reshape(x_org_shape)
        x = self.dropout1(x)
        x = self.fc1(x)      # (batch_size , sequence length , output_dim * n_components)
        if split_in_branch:
            x = x.view(*x.shape[:-1], self.output_dim, self.n_components)  # (batch_size , sequence length , output_dim , n_components)
        return x    


# DeepONet Trunk Net
class TrunkNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_components):
        super(TrunkNet, self).__init__()
        self.output_dim = output_dim
        self.n_components = n_components
        self.fc1 = nn.Linear(input_dim, trunk_internal_dim)  
        self.dropout1 = nn.Dropout(p=general_dropout)  # Dropout layer 
        # self.fc2 = nn.Linear(trunk_internal_dim, output_dim)
        self.fc2 = nn.Linear(trunk_internal_dim, trunk_internal_dim)
        self.dropout2 = nn.Dropout(p=general_dropout)  # Dropout layer 
        self.fc3 = nn.Linear(trunk_internal_dim, trunk_internal_dim) # (n_nodes_out , output_dim) 
        self.dropout3 = nn.Dropout(p=general_dropout)  # Dropout layer
        self.fc4 = nn.Linear(trunk_internal_dim, output_dim * n_components) # (n_nodes_out , output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x) 
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)                                     
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)                                     # (n_nodes_out , output_dim)
        # x = torch.relu(x) 
        if split_in_branch == False:
            x = x.view(*x.shape[:-1], self.output_dim, self.n_components)
        return x





        
# DeepONet Model
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim,hidden_temporal_size, temporal_num_layers, branch_output_dim,trunk_input_dim, trunk_output_dim , branch_n_components, trunk_n_components):
        super(DeepONet, self).__init__()
        self.trunk_n_components = trunk_n_components
        self.branch_n_components = branch_n_components
        self.branch_net_a = BranchNet_A(branch_input_dim[0], hidden_temporal_size[0], temporal_num_layers[0], branch_output_dim[0], branch_n_components)
        self.branch_net_b = BranchNet_B(branch_input_dim[1], hidden_temporal_size[1], temporal_num_layers[1], branch_output_dim[1], branch_n_components)
        self.trunk_net = TrunkNet(trunk_input_dim, trunk_output_dim, trunk_n_components)
        
        self.bc1_theta_bc_value = MDLH.normalize_data(0 , my_dataset.var0_stats, normalization_option)
        self.bc1_x_bc_value = MDLH.normalize_data(0 , my_dataset.var1_stats, normalization_option)
        self.bc1_y_bc_value = MDLH.normalize_data(0 , my_dataset.var2_stats, normalization_option)
        self.bc1_z_bc_value = MDLH.normalize_data(0 , my_dataset.var3_stats, normalization_option)
        
        self.bc2_theta_bc_value = MDLH.normalize_data(10 , my_dataset.var0_stats, normalization_option)
        
        bc2_time_scaling_array = np.concatenate([np.linspace(0, 10, 11), np.ones(90)*10])
        self.bc2_time_scaling_tensor = torch.tensor(bc2_time_scaling_array[1:], dtype=torch.float32).view(1,-1,1).to(DEVICE)

    def forward(self, load_input, strain_input ,trunk_input):
        # switch based on selected_input_option
        if selected_input_option == InputOptions.Load:
            branch_input = load_input
            branch_output = self.branch_net_a(branch_input) # (batch_size , sequence length , branch_output_dim_a, branch_n_components)
        elif selected_input_option == InputOptions.Strain:
            branch_input = strain_input
            branch_output = self.branch_net_b(branch_input) # (batch_size , sequence length , branch_output_dim_b, branch_n_components)
        elif selected_input_option == InputOptions.Load_Strain:
            branch_input_a = load_input
            branch_input_b = strain_input    
            branch_output_a = self.branch_net_a(branch_input_a) # (batch_size , sequence length , branch_output_dim_a, branch_n_components)
            branch_output_b = self.branch_net_b(branch_input_b) # (batch_size , sequence length , branch_output_dim_b, branch_n_components)
            
            branch_output = torch.cat([branch_output_a, branch_output_b], dim=-2) # (batch_size , sequence length , branch_output_dim_a + branch_output_dim_b, branch_n_components)
        
        
        # compute branch and trunk outputs
        trunk_output = self.trunk_net(trunk_input)   # (n_nodes_out , trunk_output_dim)
        
        if split_in_branch:
            NN_output =  torch.einsum('bsdc,nd->bsnc', branch_output, trunk_output)   # (batch_size , sequence length , n_nodes_out , branch_n_components)
        else:
            NN_output =  torch.einsum('bsd,ndc->bsnc', branch_output, trunk_output)
            
        if enforce_BC:
            bc_zeros_comp = torch.zeros(*NN_output.shape[:-1],1).to(DEVICE) # for all components (one dim is added)
            # enforce values at x = 0
            x_values = trunk_input[:,0]
            x_function_NN = torch.tanh((x_values+0)*10)
            x_function_NN_reshaped = x_function_NN.view(1,1,-1,1)
            if train_for_theta_only:
                bc1 = bc_zeros_comp + self.bc1_theta_bc_value
            else:
                bc1 = torch.cat([bc_zeros_comp + self.bc1_theta_bc_value, bc_zeros_comp + self.bc1_x_bc_value, bc_zeros_comp + self.bc1_y_bc_value, bc_zeros_comp + self.bc1_z_bc_value], dim=-1)
            EF_output = NN_output * x_function_NN_reshaped + bc1 * (1-x_function_NN_reshaped)
            # enforce values at y = 1
            y_values = trunk_input[:,1]
            y_function_NN = torch.tanh((y_values-1)*-10)
            y_function_NN_reshaped = y_function_NN.view(1,1,-1)
            bc_ones_comp = torch.ones(*NN_output.shape[:-1]).to(DEVICE) # for one components (No dim is added)
            time_length = EF_output.shape[1]
            bc2 = (self.bc2_time_scaling_tensor[:,:time_length,:] * x_values.view(1,1,-1))
            bc2 = MDLH.normalize_data(bc2, my_dataset.var0_stats, normalization_option)
            EF_output[...,0] = EF_output[...,0] * y_function_NN_reshaped + (bc_ones_comp * bc2)  * (1-y_function_NN_reshaped) 
            return EF_output
        else:    
            return NN_output
            
    


#_%%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_logger.log_and_print(f"DEVICE: {DEVICE}")

#_%%
# Instantiate model
model = DeepONet(branch_input_dim,hidden_temporal_size, temporal_num_layers, branch_output_dim,trunk_input_dim, trunk_output_dim, branch_n_components, trunk_n_components).to(DEVICE)

train_loss_values = []
val_loss_values = []
lr_values_history = []
# Loss and optimizer
criterion = MDLH.get_loss_criterion(loss_criterion_option)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)
starting_lr = 1e-4
scheduler_config_list = [
]
scheduler = MDLH.CustomLRScheduler(optimizer,scheduler_config_list)

#_%%
my_logger.log_and_print(model.__str__())

#_%%
# print the number of trainable parameters in the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
my_logger.log_and_print(f'Number of trainable parameters: {num_params:,}')

#_%%
training_data_dict = {
    'train_loss_values': train_loss_values,
    'val_loss_values': val_loss_values,
    'lr_values_history': lr_values_history,
    'global_trunk_input_tensor': global_trunk_input_tensor,
    'tst_trunk_input_tensor': tst_trunk_input_tensor,
    'my_dataset': my_dataset,
    'tst_dataset': tst_dataset,
    'DEVICE': DEVICE,
    'model': model,
    'train_dataloader': train_dataloader,
    'val_dataloader': val_dataloader,
    'tst_dataloader': tst_dataloader,
    'criterion': criterion,
    'optimizer': optimizer,
    'use_scheduler': use_scheduler,
    'scheduler': scheduler,
    'last_model_path': last_model_path,
    'best_model_path_train': best_model_path_train,
    'best_model_path_val': best_model_path_val,
    'node_indices': node_indices,
    'tst_node_indices': tst_node_indices,
    'trial_save_number': trial_save_number,
}

#_%%
# load the previous model to extend training
if extend_on_previous_model:
    MDLH.load_model_checkpoint(training_data_dict, previous_model_path)

#_%%
for epochs in epochs_sessions_list:
    # train the model
    MDLH.train_for_num_epochs(epochs, training_data_dict)
    MDLH.plot_and_print_loss_values(training_data_dict)
    MDLH.plot_and_print_lr_values(training_data_dict)
    # inspect the results
    current_epochs = len(training_data_dict['train_loss_values'])
    current_epochs_zero_padded = str(current_epochs).zfill(5)
    MDLH.inspect_results_for_model_and_dataloader(training_data_dict, None, val_dataloader     , f'E{current_epochs_zero_padded}_B_val_data'  , mode = 'train')
    MDLH.inspect_results_for_model_and_dataloader(training_data_dict, None, train_dataloader_NS, f'E{current_epochs_zero_padded}_C_train_data', mode = 'train')
    if plot_results_for_testing:
        MDLH.inspect_results_for_model_and_dataloader(training_data_dict, None, tst_dataloader     , f'E{current_epochs_zero_padded}_A_test_data', mode = 'test')
#_%%
# clear cuda cache
torch.cuda.empty_cache()
# call garabage collector
gc.collect()

#_%%
MDLH.inspect_results_for_model_and_dataloader(training_data_dict, best_model_path_train, train_dataloader_NS, 'D_best_train_train_data', mode = 'train')
MDLH.inspect_results_for_model_and_dataloader(training_data_dict, best_model_path_train, val_dataloader     , 'C_best_train_val_data'  , mode = 'train')
MDLH.inspect_results_for_model_and_dataloader(training_data_dict, best_model_path_val  , val_dataloader     , 'B_best_val_val_data'    , mode = 'train')
if plot_results_for_testing:
    MDLH.inspect_results_for_model_and_dataloader(training_data_dict, best_model_path_val  , tst_dataloader     , 'A_best_val_test_data', mode = 'test')
#_%%
# compute end time
overall_end_time = time.time()
overall_duration = overall_end_time - overall_starting_time
my_logger.log_and_print(f"Overall duration: {overall_duration/(60*60)} hours")
