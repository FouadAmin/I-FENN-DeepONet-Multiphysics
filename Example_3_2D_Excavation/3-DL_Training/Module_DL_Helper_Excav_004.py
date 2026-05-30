import torch
import h5py
import numpy as np
import pickle
from torch.utils.data import Dataset as TorchDataset
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torch.optim.lr_scheduler import _LRScheduler

import pyvista as pv
from scipy.ndimage import zoom

def h5_read_as_np(file_path, key='data'):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        return np.array(h5_file[key])
    
def get_group_start_end_indices(load_case_index, group_increment):
    group_index = load_case_index // group_increment
    group_start = group_index * group_increment
    group_end = group_start + group_increment
    local_index = load_case_index - group_start
    return group_index, group_start, group_end, local_index    

def read_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def normalize_data(data, stats, normalization_option):
    # 0: no normalization, 1: normalize between 0 and 1, 2: normalize between -1 and 1 3: standardization 4: divide by max value 5: divide by std
    if normalization_option == 1:
        norm_data = (data - stats['min']) / (stats['max'] - stats['min'])
    elif normalization_option == 2:
        norm_data = 2 * (data - stats['min']) / (stats['max'] - stats['min']) - 1
    elif normalization_option == 3:
        norm_data = (data - stats['mean']) / stats['std']
    elif normalization_option == 4:     
        norm_data = data / max(abs(stats['max']), abs(stats['min']))
    elif normalization_option == 5:
        norm_data = data / stats['std']
    else:
        global my_logger
        my_logger.log_and_print('Warning no normalization')
    return norm_data    

def denormalize_data(data, stats, normalization_option):
    # 0: no normalization, 1: normalize between 0 and 1, 2: normalize between -1 and 1 3: standardization 4: divide by max value 5: divide by std
    if normalization_option == 1:
        denorm_data = data * (stats['max'] - stats['min']) + stats['min']
    elif normalization_option == 2:
        denorm_data = (data + 1) * (stats['max'] - stats['min']) / 2 + stats['min']
    elif normalization_option == 3:
        denorm_data = data * stats['std'] + stats['mean']
    elif normalization_option == 4:     
        denorm_data = data * max(abs(stats['max']), abs(stats['min']))
    elif normalization_option == 5:
        denorm_data = data * stats['std']
    else:
        global my_logger
        my_logger.log_and_print('Warning no normalization')
    return denorm_data


def get_loss_criterion(loss_option):
    # 1 : MSELoss('mean') 2 : L1Loss('mean') 3 : SmoothL1Loss('mean') 4 : MSELoss('sum') 5 : L1Loss('sum') 6 : SmoothL1Loss('sum') 7 : CustomLoss (L2) 8 : CustomLoss (Normalized L2)
    if loss_option == 1:
        criterion = torch.nn.MSELoss(reduction='mean') # default
    elif loss_option == 2:
        criterion = torch.nn.L1Loss(reduction='mean') # default
    elif loss_option == 3:
        criterion = torch.nn.SmoothL1Loss(reduction='mean') # default
    elif loss_option == 4:
        criterion = torch.nn.MSELoss(reduction='sum')
    elif loss_option == 5:
        criterion = torch.nn.L1Loss(reduction='sum')
    elif loss_option == 6:
        criterion = torch.nn.SmoothL1Loss(reduction='sum')
    elif loss_option == 7:
        # L2 Norm
        # sqrt of sum of squared differences
        class CustomLoss(torch.nn.Module):
            def __init__(self):
                super(CustomLoss, self).__init__()

            def forward(self, true_data, predicted_data):
                loss = torch.sqrt(torch.sum((true_data - predicted_data) ** 2))
                return loss
        criterion = CustomLoss()    
    elif loss_option == 8:
        # Normalized L2 Norm
        # sqrt of sum of squared differences
        class CustomLoss(torch.nn.Module):
            def __init__(self):
                super(CustomLoss, self).__init__()

            def forward(self, true_data, predicted_data):
                loss = torch.sqrt(torch.sum((true_data - predicted_data) ** 2))/torch.sqrt(torch.sum((true_data) ** 2))
                return loss
        criterion = CustomLoss()
    else:
        raise ValueError('Invalid loss_option')    
    return criterion


def map_array_from_grid_to_list(array, map_grid_to_list_org, skip_i, skip_j):
    dim_i = array.shape[-3]
    dim_j = array.shape[-2]
    dim_k = array.shape[-1]

    new_list_index = 0
    map_grid_to_list_new = np.full((dim_i, dim_j), int(-1e5), dtype=int)
    for i in range(dim_i):
        for j in range(dim_j):
            i_list = map_grid_to_list_org[i,j]
            if i_list > -1 and i % skip_i == 0 and j % skip_j == 0:
                map_grid_to_list_new[i,j] = new_list_index
                new_list_index += 1                    
    
    new_list_length = new_list_index 
    # new mapped array that have the same shape as the original array but the last three dimensions removed and replaced with the new_list_length,dim_k
    original_shape = array.shape
    new_shape = list(original_shape)[:-3]
    new_shape.append(new_list_length)
    new_shape.append(dim_k)
    new_shape = tuple(new_shape)
    
    # check if array is a pytorch tensor
    if isinstance(array, torch.Tensor):
        mapped_array = torch.full(new_shape, np.nan, dtype=torch.float32)
    else:
        mapped_array = np.full(new_shape, np.nan, dtype=np.float64)
    
    map_list_to_grid_new = np.full((new_list_length, 2), int(-1e5), dtype=int)
    
    for i in range(dim_i):
        for j in range(dim_j):
            i_list_new = map_grid_to_list_new[i,j]
            if i_list_new > -1:
                mapped_array[...,i_list_new,:] = array[...,i,j,:]
                map_list_to_grid_new[i_list_new,0] = i
                map_list_to_grid_new[i_list_new,1] = j
                
    return mapped_array, map_grid_to_list_new, map_list_to_grid_new

def map_array_from_list_to_grid(array, map_list_to_grid,dim_i, dim_j):
    if dim_i is None:
        max_i_index = np.max(map_list_to_grid[:,0])
        dim_i = int(np.round(max_i_index + 1))
        
    if dim_j is None:
        max_j_index = np.max(map_list_to_grid[:,1])
        dim_j = int(np.round(max_j_index + 1))

    dim_k = array.shape[-1]

    length_list = array.shape[-2]
    original_shape = array.shape
    new_shape = list(original_shape)[:-2]
    new_shape.append(dim_i)
    new_shape.append(dim_j)
    new_shape.append(dim_k)
    new_shape = tuple(new_shape)
    
    # check if array is a pytorch tensor
    if isinstance(array, torch.Tensor):
        mapped_array = torch.full(new_shape, np.nan, dtype=torch.float32)
    else:
        mapped_array = np.full(new_shape, np.nan, dtype=np.float64)
        
    for i_list in range(length_list):
        i_grid = map_list_to_grid[i_list,0]
        j_grid = map_list_to_grid[i_list,1]
        if i_grid > -1 and j_grid > -1:
            mapped_array[...,i_grid,j_grid,:] = array[...,i_list,:]
    return mapped_array

# define a class for the scheduler
class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, scheduler_config_list, last_epoch=-1):
        self.scheduler_config_list = scheduler_config_list
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        for scheduler_config in self.scheduler_config_list:
            sch_type = scheduler_config[0]
            start_ep = scheduler_config[1]
            end_ep = scheduler_config[2]
            if start_ep <= current_epoch < end_ep:
                if sch_type == 'linear':
                    start_lr = scheduler_config[3]
                    end_lr = scheduler_config[4]
                    lr = start_lr + (end_lr - start_lr) * (current_epoch - start_ep) / (end_ep - start_ep)
                elif sch_type == 'const':
                    lr = scheduler_config[3]
                else:
                    raise ValueError(f'Unknown scheduler type: {type}')
                return [lr for _ in self.optimizer.param_groups]

        # if current_epoch is out of the range of all schedulers, return the last learning rate
        return [group['lr'] for group in self.optimizer.param_groups]

# define a class for the dataset
class MyDataset(TorchDataset):
    def __init__(self, dl_settings, mode):
        self.mode = mode
        self.load_folder_path = dl_settings['load_folder_path']
        self.output_folder_path = dl_settings['output_folder_path']
        self.strain_folder_path = dl_settings['strain_folder_path']
        self.load_file_name = dl_settings['load_file_name']
        self.output_file_name = dl_settings['output_file_name']
        self.strain_file_name = dl_settings['strain_file_name']
        self.coordinates_file_name = dl_settings['coordinates_file_name']
        self.map_nodes_grid_to_list_file_name = dl_settings['map_nodes_grid_to_list_file_name']
        self.map_cells_grid_to_list_file_name = dl_settings['map_cells_grid_to_list_file_name']
        self.LC_start = dl_settings['LC_start']
        self.LC_end = dl_settings['LC_end']
        self.slicing_settings = dl_settings['slicing_settings']
        self.normalization_option = dl_settings['normalization_option']
        self.normalization_option_coords = dl_settings['normalization_option_coords']
        self.max_items_saved = dl_settings['max_items_saved']
        self.read_all_the_file_once = dl_settings['read_all_the_file_once']
        self.data_info = []
        self.data_per_group = {}
        self.original_load_grid_shape = None
        self.original_strain_grid_shape = None
        self.original_output_grid_shape = None
       
        
        self.load_stats = read_pickle_data(f'{self.load_folder_path}/{self.load_file_name}stats.pkl')
        self.strain_stats = read_pickle_data(f'{self.strain_folder_path}/{self.strain_file_name}stats.pkl')
        self.var0_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var0.pkl')
        self.var1_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var1.pkl')
        self.var2_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var2.pkl')
        
        self.original_coordinates_array_grid = h5_read_as_np(f'{self.output_folder_path}/{self.coordinates_file_name}0-100.h5')
        self.coord_stats_var0 = read_pickle_data(f'{self.output_folder_path}/{self.coordinates_file_name}stats_var0.pkl')
        self.coord_stats_var1 = read_pickle_data(f'{self.output_folder_path}/{self.coordinates_file_name}stats_var1.pkl')
        self.load_gen_data = np.load(f'{self.load_folder_path}/{self.load_file_name}.npy')
        
        self.map_nodes_grid_to_list_original = h5_read_as_np(f'{self.output_folder_path}/{self.map_nodes_grid_to_list_file_name}.h5')
        self.map_cells_grid_to_list_original = h5_read_as_np(f'{self.output_folder_path}/{self.map_cells_grid_to_list_file_name}.h5')
        
        self.map_nodes_grid_to_list_current = None
        self.map_cells_grid_to_list_current = None
        self.map_nodes_list_to_grid_current = None
        self.map_cells_list_to_grid_current = None
        
        for dataset_index, global_index in enumerate(range(self.LC_start, self.LC_end)):
            group_index, group_start, group_end, local_index = get_group_start_end_indices(global_index, 100)
            load_file_path = f'{self.load_folder_path}/{self.load_file_name}{group_start}-{group_end}.h5'
            strain_file_path = f'{self.strain_folder_path}/{self.strain_file_name}{group_start}-{group_end}.h5'
            output_file_path = f'{self.output_folder_path}/{self.output_file_name}{group_start}-{group_end}.h5'
            
            self.data_info.append((load_file_path, strain_file_path, output_file_path, local_index, dataset_index, global_index))
            
            if group_index not in self.data_per_group:
                self.data_per_group[group_index] = []
                
            self.data_per_group[group_index].append((load_file_path, strain_file_path, output_file_path, local_index, dataset_index, global_index))  
            
        assert len(self.data_info) == (self.LC_end - self.LC_start), 'Error : data_info length should be equal to (LC_end - LC_start)'    
            
        if self.max_items_saved >= (self.LC_end - self.LC_start):
            self.pre_load_all_data()
            

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        global main_cashed_items_dict
        global test_cashed_items_dict
        
        if self.mode == 'test':
            current_cashed_items_dict = test_cashed_items_dict
        else:
            current_cashed_items_dict = main_cashed_items_dict
            
        if idx in current_cashed_items_dict:
            load_array, strain_array, output_array = current_cashed_items_dict[idx]
        else:
            load_array, strain_array, output_array = self.get_new_item(idx)
        
        global use_torch_multiprocessing_manager
        
        if use_torch_multiprocessing_manager == True: # data is stored as numpy arrays in the shared dictionary
            # convert to tensors
            load_tensor = torch.tensor(load_array, dtype=torch.float32)
            strain_tensor = torch.tensor(strain_array, dtype=torch.float32)
            output_tensor = torch.tensor(output_array, dtype=torch.float32)
            return load_tensor, strain_tensor, output_tensor    
        else: # data is stored as tensors in the shared dictionary => no need to convert
            return load_array, strain_array, output_array
     
    def post_process_loaded_data(self, load_array, strain_array, output_array):
        
        strain_start = self.slicing_settings['point_strain_start']
        strain_skip = self.slicing_settings['point_strain_skip']
        strain_end = self.slicing_settings['point_strain_end']
        output_start = self.slicing_settings['point_output_start']
        output_skip = self.slicing_settings['point_output_skip']
        output_end = self.slicing_settings['point_output_end']
        
        global train_for_pressure_only
        
        if train_for_pressure_only == True:
            # only train for pressure
            output_array = output_array[..., 0:1]
        
        
        # Note: By converting tensors to numpy arrays before storing them in the shared dictionary 
        # and converting them back to tensors when retrieving them, you can avoid issues related to 
        # tensor serialization and deserialization in multiprocessing.
        
        global use_torch_multiprocessing_manager
        
        if use_torch_multiprocessing_manager == False: # better for performance to avoid converting to tensors on loading
            # convert to tensors
            load_array = torch.tensor(load_array, dtype=torch.float32)
            strain_array = torch.tensor(strain_array, dtype=torch.float32)
            output_array = torch.tensor(output_array, dtype=torch.float32)
        
        
        # store the original shape
        self.original_load_grid_shape = load_array.shape
        self.original_strain_grid_shape = strain_array.shape
        self.original_output_grid_shape = output_array.shape
        
        if self.slicing_settings['point_strain_flatten']:
            strain_array, self.map_cells_grid_to_list_current, self.map_cells_list_to_grid_current = map_array_from_grid_to_list(strain_array, self.map_cells_grid_to_list_original, strain_skip[0], strain_skip[1])    
            # NOTE: mapping requires the last dimension to be n features which is 1 in this case
            # so we need to squeeze the last dimension after mapping
            strain_array = strain_array.squeeze(-1)
            
        if self.slicing_settings['point_output_flatten']:
            output_array, self.map_nodes_grid_to_list_current, self.map_nodes_list_to_grid_current = map_array_from_grid_to_list(output_array, self.map_nodes_grid_to_list_original, output_skip[0], output_skip[1])

        load_array = normalize_data(load_array, self.load_stats, self.normalization_option)

        strain_array = normalize_data(strain_array, self.strain_stats, self.normalization_option)

        output_array[..., 0] = normalize_data(output_array[..., 0], self.var0_stats, self.normalization_option)
        if train_for_pressure_only == False:
            output_array[..., 1] = normalize_data(output_array[..., 1], self.var1_stats, self.normalization_option)
            output_array[..., 2] = normalize_data(output_array[..., 2], self.var2_stats, self.normalization_option)
        
        return load_array, strain_array, output_array
    
    def unflatten_tensor(self, tensor, original_shape, batch_included):
        if batch_included:
            batch_size = tensor.shape[0]
            tensor = tensor.reshape(batch_size, *original_shape)
        else:
            tensor = tensor.reshape(original_shape)
        return tensor
     
    def denormalize_load_tensor(self, load_tensor):
        denormalized_load_tensor = denormalize_data(load_tensor, self.load_stats, self.normalization_option)
        return denormalized_load_tensor
    
    def denormalize_strain_tensor(self, strain_tensor):
        denormalized_strain_tensor = denormalize_data(strain_tensor, self.strain_stats, self.normalization_option)
        return denormalized_strain_tensor
    
    def normalize_strain_tensor(self, strain_tensor):
        normalized_strain_tensor = normalize_data(strain_tensor, self.strain_stats, self.normalization_option)
        return normalized_strain_tensor
    
    def normalize_coord_tensor(self, coord_tensor):
        # check if numpy or tensor
        if isinstance(coord_tensor, torch.Tensor):
            normalized_coord_tensor = torch.zeros_like(coord_tensor)
        else:
            normalized_coord_tensor = np.zeros_like(coord_tensor)
        
        normalized_coord_tensor[..., 0] = normalize_data(coord_tensor[..., 0], self.coord_stats_var0, self.normalization_option_coords)
        normalized_coord_tensor[..., 1] = normalize_data(coord_tensor[..., 1], self.coord_stats_var1, self.normalization_option_coords)
        return normalized_coord_tensor
    
    def denormalize_coord_tensor(self, coord_tensor):
        # check if numpy or tensor
        if isinstance(coord_tensor, torch.Tensor):
            denormalized_coord_tensor = torch.zeros_like(coord_tensor)
        else:
            denormalized_coord_tensor = np.zeros_like(coord_tensor)
        
        denormalized_coord_tensor[..., 0] = denormalize_data(coord_tensor[..., 0], self.coord_stats_var0, self.normalization_option_coords)
        denormalized_coord_tensor[..., 1] = denormalize_data(coord_tensor[..., 1], self.coord_stats_var1, self.normalization_option_coords)
        return denormalized_coord_tensor
    
    def denormalize_output_tensor(self, output_tensor):
        global train_for_pressure_only
        # check if numpy or tensor
        if isinstance(output_tensor, torch.Tensor):
            denormalized_output_tensor = torch.zeros_like(output_tensor)
        else:
            denormalized_output_tensor = np.zeros_like(output_tensor)    
        
        denormalized_output_tensor[..., 0] = denormalize_data(output_tensor[..., 0], self.var0_stats, self.normalization_option)
        if train_for_pressure_only == False:
            denormalized_output_tensor[..., 1] = denormalize_data(output_tensor[..., 1], self.var1_stats, self.normalization_option)
            denormalized_output_tensor[..., 2] = denormalize_data(output_tensor[..., 2], self.var2_stats, self.normalization_option)
        
        return denormalized_output_tensor
     
    def pre_load_all_data(self):
        global main_cashed_items_dict
        global test_cashed_items_dict
        global my_logger

        if self.mode == 'test':
            current_cashed_items_dict = test_cashed_items_dict
        else:
            current_cashed_items_dict = main_cashed_items_dict

        my_logger.log_and_print(f'..... Preloading all data ..... started at {time.ctime()}')
        start_time = time.time()
        
        load_time_start = self.slicing_settings['load_time_start']
        load_time_skip = self.slicing_settings['load_time_skip']
        load_time_end = self.slicing_settings['load_time_end']
        strain_time_start = self.slicing_settings['strain_time_start']
        strain_time_skip = self.slicing_settings['strain_time_skip']
        strain_time_end = self.slicing_settings['strain_time_end']
        
        # loop over keys in data_per_group
        for group_index in self.data_per_group:
            required_data_in_group = self.data_per_group[group_index]
            load_file_path, strain_file_path , output_file_path, _ , _, _ = required_data_in_group[0]
            
            with h5py.File(output_file_path, 'r') as output_file, h5py.File(strain_file_path, 'r') as strain_file:
                if self.read_all_the_file_once:
                    my_logger.log_and_print(f'Loading all at once at group {group_index}')
                    strain_data_all = np.array(strain_file['data'])
                    output_data_all = np.array(output_file['data'])
                    for _,_,_, local_index, dataset_index, global_index in required_data_in_group:
                        load_data = self.load_gen_data[global_index,load_time_start:load_time_end:load_time_skip,:]
                        strain_data = strain_data_all[local_index, strain_time_start:strain_time_end:strain_time_skip, :, :, :]
                        output_data = output_data_all[local_index, load_time_start:load_time_end:load_time_skip, :, :, :]
                        load_array, strain_array, output_array = self.post_process_loaded_data(load_data, strain_data, output_data)
                        current_cashed_items_dict[dataset_index] = (load_array, strain_array, output_array)
                    
                else:    
                    my_logger.log_and_print(f'Loading index by index at group {group_index}')
                    for _,_,_, local_index, dataset_index, global_index in required_data_in_group:
                        load_data = self.load_gen_data[global_index,load_time_start:load_time_end:load_time_skip,:]
                        strain_data = strain_file['data'][local_index, strain_time_start:strain_time_end:strain_time_skip, :, :, :]
                        output_data = output_file['data'][local_index, load_time_start:load_time_end:load_time_skip, :, :, :]
                        load_array, strain_array, output_array = self.post_process_loaded_data(load_data, strain_data, output_data)
                        current_cashed_items_dict[dataset_index] = (load_array, strain_array, output_array)
        
        assert len(current_cashed_items_dict) == len(self.data_info), 'Error : All data not loaded correctly in pre_load_all_data'
        
        end_time = time.time()            
        my_logger.log_and_print(f'..... Preloading all data ..... ended at {time.ctime()} and took {(end_time - start_time)/60} minutes')    
                
        
    def get_new_item(self, idx):
        global main_cashed_items_dict
        global test_cashed_items_dict
        
        
        load_file_path, strain_file_path, output_file_path, local_index, dataset_index, global_index = self.data_info[idx]
        
        assert dataset_index == idx , 'Error : dataset_index and idx should be the same'
        
        load_time_start = self.slicing_settings['load_time_start']
        load_time_skip = self.slicing_settings['load_time_skip']
        load_time_end = self.slicing_settings['load_time_end']
        strain_time_start = self.slicing_settings['strain_time_start']
        strain_time_skip = self.slicing_settings['strain_time_skip']
        strain_time_end = self.slicing_settings['strain_time_end']
        
        load_data = self.load_gen_data[global_index,load_time_start:load_time_end:load_time_skip,:]
        
        with h5py.File(strain_file_path, 'r') as h5_file:
            strain_data = h5_file['data'][local_index, strain_time_start:strain_time_end:strain_time_skip, :, :, :]
        with h5py.File(output_file_path, 'r') as h5_file:
            output_data = h5_file['data'][local_index, load_time_start:load_time_end:load_time_skip, :, :, :]
        
        load_array, strain_array, output_array = self.post_process_loaded_data(load_data, strain_data, output_data)
        
        if self.mode == 'test':
            current_cashed_items_dict = test_cashed_items_dict
        else:
            current_cashed_items_dict = main_cashed_items_dict
            
        if len(current_cashed_items_dict) < self.max_items_saved:
            current_cashed_items_dict[idx] = (load_array, strain_array, output_array)
        
        return load_array, strain_array, output_array
    
    def get_mapped_back_coordinates_array_grid(self):
        sliced_coordinates_list = self.get_coordinates_array_list(return_as_tensor=False, normalize=False)
        sliced_coordinates_grid = map_array_from_list_to_grid(sliced_coordinates_list, self.map_nodes_list_to_grid_current, self.original_coordinates_array_grid.shape[0], self.original_coordinates_array_grid.shape[1])
        return sliced_coordinates_grid
    
    def get_coordinates_array_list(self, return_as_tensor, normalize):
        
        output_start = self.slicing_settings['point_output_start']
        output_skip = self.slicing_settings['point_output_skip']
        output_end = self.slicing_settings['point_output_end']
        
        coordinates_array_list = self.original_coordinates_array_grid.copy()
        
        if self.slicing_settings['point_output_flatten']:
            sliced_coordinates_list, _, _ = map_array_from_grid_to_list(coordinates_array_list, self.map_nodes_grid_to_list_original, output_skip[0], output_skip[1])
        
        
        if normalize:
            sliced_coordinates_list = self.normalize_coord_tensor(sliced_coordinates_list)
                     
        if return_as_tensor:
            return torch.tensor(sliced_coordinates_list, dtype=torch.float32)
        else:
            return sliced_coordinates_list
    
    def get_index_of_coordinates_list(self, coordinates_list):
        sliced_coordinates_list = self.get_coordinates_array_list(return_as_tensor=False, normalize=False)
        index_list = []
        for coordinates in coordinates_list:
            # find the nearest point in the sliced_coordinates_array_per_id
            index = np.argmin(np.linalg.norm(sliced_coordinates_list - coordinates, axis=1))
            index_list.append(index)
            
        return index_list    

def save_model_checkpoint(training_data_dict, checkpoint_path):    
    global my_logger
    model = training_data_dict['model']
    optimizer = training_data_dict['optimizer']
    scheduler = training_data_dict['scheduler']
    train_loss_values = training_data_dict['train_loss_values']
    val_loss_values = training_data_dict['val_loss_values']
    lr_values_history = training_data_dict['lr_values_history']
    epoch = len(train_loss_values)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_values': train_loss_values,
        'val_loss_values': val_loss_values,
        'lr_values_history': lr_values_history
    }
    torch.save(checkpoint, checkpoint_path)
    
def load_model_checkpoint(training_data_dict, checkpoint_path):
    model = training_data_dict['model']
    optimizer = training_data_dict['optimizer']
    scheduler = training_data_dict['scheduler']
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    training_data_dict['train_loss_values'] = checkpoint['train_loss_values']
    training_data_dict['val_loss_values'] = checkpoint['val_loss_values']
    training_data_dict['lr_values_history'] = checkpoint['lr_values_history']
    
    global my_logger
    my_logger.add_to_wait_list(f'###############################################################################')
    my_logger.add_to_wait_list(f'Checkpoint loaded from {checkpoint_path} at epoch {checkpoint["epoch"]}')
    my_logger.add_to_wait_list(f'###############################################################################')
    my_logger.log_and_print_the_wait_list()
    
    return   
    
    
def train_for_num_epochs(num_epochs,training_data_dict):
    train_loss_values = training_data_dict['train_loss_values']
    val_loss_values = training_data_dict['val_loss_values']
    lr_values_history = training_data_dict['lr_values_history']
    global_trunk_input_tensor = training_data_dict['global_trunk_input_tensor']
    DEVICE = training_data_dict['DEVICE']
    model = training_data_dict['model']
    train_dataloader = training_data_dict['train_dataloader']
    val_dataloader = training_data_dict['val_dataloader']
    criterion = training_data_dict['criterion']
    optimizer = training_data_dict['optimizer']
    use_scheduler = training_data_dict['use_scheduler']
    scheduler = training_data_dict['scheduler']
    last_model_path = training_data_dict['last_model_path']
    best_model_path_train = training_data_dict['best_model_path_train']
    best_model_path_val = training_data_dict['best_model_path_val']
    
    global my_logger
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    my_logger.add_to_wait_list(f'Training started for num_epochs: {num_epochs} at {time.ctime()}')
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    my_logger.log_and_print_the_wait_list()
        
    st = time.time() # total training start time
    
    n_previous_epochs = len(train_loss_values)
    min_val_loss = float('inf')
    min_train_loss = float('inf')
    new_best_train_count = 0
    new_best_val_count = 0
    
    trunk_input = global_trunk_input_tensor # (coords are the same for all batches)
    trunk_input = trunk_input.to(DEVICE)
    original_logger_wait_list_length = my_logger.max_wait_list_length
    for epoch in range(num_epochs):
        if epoch < 2:
            my_logger.max_wait_list_length = 1
        else:
            my_logger.max_wait_list_length = original_logger_wait_list_length
        ## TRAINING
        model.train()  # Set the model to training mode
        st_e = time.time() # epoch training start time
        weighted_epoch_loss = 0
        weights_sum = 0
        for i, (load_input, strain_input, true_output) in enumerate(train_dataloader):
            load_input = load_input.to(DEVICE)
            strain_input = strain_input.to(DEVICE)
            true_output = true_output.to(DEVICE)
            
            # Forward pass
            predictions = model(load_input, strain_input, trunk_input)
            loss = criterion(predictions, true_output)
            weighted_epoch_loss += loss.item() * load_input.shape[0]
            weights_sum += load_input.shape[0]
            
            # Backward pass
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        lr_values_history.append(current_lr)    
        
        # Step the scheduler
        if use_scheduler:
            scheduler.step()
        
        loss_value = weighted_epoch_loss/weights_sum
        train_loss_values.append(loss_value)
        et_e = time.time() # epoch training end time
        if loss_value < min_train_loss:
            new_best_train_count += 1
        if (loss_value < 0.95 * min_train_loss) or (new_best_train_count >= 10) or (epoch + n_previous_epochs < 10):
            new_best_train_count = 0
            min_train_loss = loss_value
            save_model_checkpoint(training_data_dict, best_model_path_train)
            my_logger.add_to_wait_list(f'Epoch [{epoch+1+n_previous_epochs:4.0f}/{num_epochs+n_previous_epochs:4.0f}], Loss: {loss_value:.3e} , execution time: {(et_e - st_e)/60:.3f} minutes || New Best ....')
        elif (epoch+1)%50 == 0:
            my_logger.add_to_wait_list(f'Epoch [{epoch+1+n_previous_epochs:4.0f}/{num_epochs+n_previous_epochs:4.0f}], Loss: {loss_value:.3e} , execution time: {(et_e - st_e)/60:.3f} minutes')    
            
            
        ## VALIDATION
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            weighted_val_loss = 0
            weights_sum_val = 0
            for i, (load_input, strain_input, true_output) in enumerate(val_dataloader):
                load_input = load_input.to(DEVICE)
                strain_input = strain_input.to(DEVICE)
                true_output = true_output.to(DEVICE)
                predictions = model(load_input, strain_input, trunk_input)
                loss = criterion(predictions, true_output)
                weighted_val_loss += loss.item() * load_input.shape[0]
                weights_sum_val += load_input.shape[0]
            val_loss_value = weighted_val_loss/weights_sum_val
            val_loss_values.append(val_loss_value)
            if val_loss_value < min_val_loss:
                new_best_val_count += 1
            if (val_loss_value < 0.95 * min_val_loss) or (new_best_val_count >= 10):
                new_best_val_count = 0
                min_val_loss = val_loss_value
                save_model_checkpoint(training_data_dict, best_model_path_val)
                my_logger.add_to_wait_list(f'Validation Loss: {val_loss_value:.3e} || New Best ....')
            elif (epoch+1)%50 == 0:
                my_logger.add_to_wait_list(f'Validation Loss: {val_loss_value:.3e}')
            
    my_logger.log_the_wait_list()      
            
    et = time.time() # total training end time
    my_logger.log_and_print(f'Total training execution time: {(et - st)/60:.2f} minutes')
    save_model_checkpoint(training_data_dict, last_model_path)
    


def plot_and_print_loss_values(training_data_dict):
    global my_logger
    train_loss_values = training_data_dict['train_loss_values']
    val_loss_values = training_data_dict['val_loss_values']
    n_epoch_finished = len(train_loss_values)
    
    if n_epoch_finished == 0:
        return
    
    my_logger.add_to_wait_list(f'Min Training Loss {min(train_loss_values):.3e} after {n_epoch_finished} epochs')
    my_logger.add_to_wait_list(f'Min Validation Loss {min(val_loss_values):.3e} after {n_epoch_finished} epochs')
    # Create a new figure
    fig = plt.figure()
    plt.plot(train_loss_values, label='train')
    plt.plot(val_loss_values, label='val')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Values')    
    # save the plot
    global plots_save_folder
    plt.savefig(f'./{plots_save_folder}/a-loss_values_after_{n_epoch_finished}_epochs.png')
    
    plt.close('all')   
    
    my_logger.log_and_print_the_wait_list()
    
    
def plot_and_print_lr_values(training_data_dict):
    global my_logger
    lr_values_history = training_data_dict['lr_values_history']
    n_epoch_finished = len(lr_values_history)
    
    if n_epoch_finished == 0:
        return
    
    my_logger.add_to_wait_list(f'Min Learning Rate {min(lr_values_history):.3e} after {n_epoch_finished} epochs')
    
    # Create a new figure
    fig = plt.figure()
    plt.plot(lr_values_history, label='learning rate')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate History')    
    plt.tight_layout()  # This automatically adjusts the spacing

    # save the plot
    global plots_save_folder
    plt.savefig(f'./{plots_save_folder}/a-lr_values_for_{n_epoch_finished}_epochs.png')
    
    plt.close('all')   
    
    my_logger.log_and_print_the_wait_list()    

def plot_true_vs_prediction(true_output, predictions, history_index, node_index, node_coords, description, trial_save_number, node_periority_index):

    # N subplots equal last dimension size
    n_components = true_output.shape[-1]

    # Make subplots for each output
    if n_components == 1:
        fig_width = 15/4
    else:
        fig_width = 15
    fig, axs = plt.subplots(1, n_components, figsize=(fig_width, 5))
    alpha_tr = 1.0
    alpha_pr = 0.7
    
    colors = ['black', 'red', 'green', 'blue']
    labels = ['pressure', 'ux', 'uy', 'uz']
    
    if n_components == 1:
        axs = [axs]
    
    for i in range(n_components):
        axs[i].plot(true_output[history_index,:,node_index,i].flatten(), label='True', alpha=alpha_tr, color=colors[i], linewidth=2, linestyle='solid')
        axs[i].plot(predictions[history_index,:,node_index,i].flatten(), label='Predictions', alpha=alpha_pr, color=colors[i], linewidth=2, linestyle='none',marker='x')
        axs[i].set_title(labels[i])
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[i].legend()
            
    # add title for the whole figure
    title_line0 = f'{trial_save_number}'
    title_line1 = f'{description}'
    title_line2 = f'load case: {history_index} | node: {node_index}'
    title_line3 = f'coord: ({node_coords[0]:.3f},{node_coords[1]:.3f})'
    
    if n_components == 1:
        # make suptitle in two lines
        fig.suptitle(f'{title_line0} \n {title_line1} \n {title_line2} \n {title_line3}' ,  fontsize=12)
    else:
        fig.suptitle(f'{title_line0} | {title_line1} | {title_line2} | {title_line3}' ,  fontsize=12)
            
    plt.tight_layout()  # This automatically adjusts the spacing
    
    # save the plot
    global plots_save_folder
    plt.savefig(f'./{plots_save_folder}/c-predictions_{description}_hist_{history_index}_node_{node_periority_index}==>{node_index}.png')
    plt.close('all')
    
def plot_true_vs_prediction_slice(true_output, predictions, history_index, description, trial_save_number, plots_save_folder):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*changed.*")
    plt.rc('font', family= 'serif')  # 'Times New Roman'
    plt.rc('font', size=14)
    # know the dimensions
    n_time_steps = true_output.shape[0]
    angle_dim = true_output.shape[1]
    r_dim = true_output.shape[2]
    z_dim = true_output.shape[3]
    n_components = true_output.shape[-1]
    
    component_labels = [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$']
    sub_plot_no_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)']
    
    true_slices = []
    pred_slices = []
    slices_labels = []
    slices_details = []
    axes_labels = []
    save_indices = []
    
    # Slices 
    # t-x slices (line in space)
    si=0
    for iz,z_index in enumerate([z_dim-1]):
        for ir, r_index in enumerate([r_dim // 2]):
            si+=1
            relative_z_index = round((z_index+1)/z_dim,1)
            relative_r_index = round((r_index+1)/r_dim,1)
                
            true_slices.append(true_output[:, :, r_index, z_index, :])
            pred_slices.append(predictions[:, :, r_index, z_index, :])
            slices_details.append(f'at rel r={relative_r_index}, rel z={relative_z_index}')
            slices_labels.append(f't-a')
            axes_labels.append(['time [s]', r'angle/2$\pi$'])
            save_indices.append(si)
    
    # t-y slices (line in space)
    for iz,z_index in enumerate([z_dim-1]):
        for ia, a_index in enumerate([angle_dim // 2]):
            si+=1
            relative_z_index = round((z_index+1)/z_dim,1)
            relative_a_index = round(a_index/angle_dim,1)
                            
            true_slices.append(true_output[:, a_index, :, z_index, :])
            pred_slices.append(predictions[:, a_index, :, z_index, :])
            slices_details.append(f'at rel a={relative_a_index}, rel z={relative_z_index}')
            slices_labels.append(f't-r')
            axes_labels.append(['time [s]', 'r [m]'])
            save_indices.append(si)
    
    # t-z slices (line in space)
    for ir, r_index in enumerate([r_dim // 2]):
        for ia, a_index in enumerate([angle_dim // 2]):
            si+=1
            relative_r_index = round((r_index+1)/r_dim,1)
            relative_a_index = round(a_index/angle_dim,1)
                            
            true_slices.append(true_output[:, a_index, r_index, :, :])
            pred_slices.append(predictions[:, a_index, r_index, :, :])
            slices_details.append(f'at rel a={relative_a_index}, rel r={relative_r_index}')
            slices_labels.append(f't-z')
            axes_labels.append(['time [s]', 'z [m]'])
            save_indices.append(si)

    # x-y slices
    for it,t_index in enumerate([59,119]):
        for iz,z_index in enumerate([z_dim-1]):
            si+=1
            relative_z_index = round((z_index+1)/z_dim,1)
            relative_t_index = t_index + 1
            
            true_slices.append(true_output[t_index, :, :, z_index, :])
            pred_slices.append(predictions[t_index, :, :, z_index, :])
            slices_details.append(f'at rel z={relative_z_index}, t={relative_t_index}')
            slices_labels.append(f'a-r')
            axes_labels.append([r'angle/2$\pi$', 'r [m]'])
            save_indices.append(si)
    
    # x-z slices
    for it,t_index in enumerate([59,119]):
        for ir, r_index in enumerate([r_dim // 2]):
            si+=1
            relative_r_index = round((r_index+1)/r_dim,1)
            relative_t_index = t_index + 1
            
            true_slices.append(true_output[t_index, :, r_index, :, :])
            pred_slices.append(predictions[t_index, :, r_index, :, :])
            slices_details.append(f'at rel r={relative_r_index}, t={relative_t_index}')
            slices_labels.append(f'a-z')
            axes_labels.append([r'angle/2$\pi$', 'z [m]'])
            save_indices.append(si)      
            
    # y-z slices
    for it,t_index in enumerate([59,119]):
        for ia, a_index in enumerate([angle_dim // 2]):
            si+=1
            relative_a_index = round(a_index/angle_dim,1)
            relative_t_index = t_index + 1
            
            true_slices.append(true_output[t_index, a_index, :, :, :])
            pred_slices.append(predictions[t_index, a_index, :, :, :])
            slices_details.append(f'at rel a={relative_a_index}, t={relative_t_index}')
            slices_labels.append(f'r-z')
            axes_labels.append(['r [m]', 'z [m]'])
            save_indices.append(si)          
    
    n_slices = len(true_slices)
    
    for i_slice in range(n_slices):
        true_slice= true_slices[i_slice]
        pred_slice = pred_slices[i_slice]
        slice_label = slices_labels[i_slice]
        slice_details = slices_details[i_slice]
        axes_label_x = axes_labels[i_slice][0]
        axes_label_y = axes_labels[i_slice][1]
        save_index = save_indices[i_slice]
        
        data_labels = ['True', 'Predicted', '|Error|']
        data_array = [true_slice, pred_slice, np.abs(true_slice - pred_slice)]
        heat_map_type = ['jet', 'jet', 'jet']
        
        n_rows = len(data_array)
        # Make subplots for each component (columns will represent components)
        # 3 rows will represent true, predictions and difference
    
        # Create a new figure
        if n_components == 1:
            fig_width = 15/4
        else:
            fig_width = 15
            
        fig, axs = plt.subplots(n_rows,n_components,figsize=(fig_width, 10))
        
        if n_components == 1:
            axs = np.expand_dims(axs, axis=1)
        
        for i_component in range(n_components):
            for i_row in range(n_rows):
                # i_subplot = i_component * n_rows + i_row
                i_subplot = i_row * n_components + i_component
                sub_plot_no = sub_plot_no_list[i_subplot]
                slice_to_plot = data_array[i_row][:,:,i_component]
                # determine max and min values for colorbar
                current_max = np.max(slice_to_plot)
                current_min = np.min(slice_to_plot)
                if i_row == 0:
                    true_max = current_max
                    true_min = current_min
                
                if i_row == 1:
                    to_use_max = true_max
                    to_use_min = true_min
                else:
                    to_use_max = current_max
                    to_use_min = current_min    
                
                # change diplayed values on axis to range from 0 to 1
                if axes_label_x == 'time [s]':
                    x_axis_min = 0
                    x_axis_max = 60000
                    x_axis_n = 3
                elif axes_label_x == r'angle/2$\pi$':
                    x_axis_min = 0
                    x_axis_max = 1
                    x_axis_n = 3
                elif axes_label_x == 'r [m]':    
                    x_axis_min = 1
                    x_axis_max = 2
                    x_axis_n = 3
                else:
                    raise ValueError('Unknown axes_label_x')
                
                if axes_label_y == r'angle/2$\pi$':
                    y_axis_min = 0
                    y_axis_max = 1
                    y_axis_n = 3
                elif axes_label_y == 'r [m]':
                    y_axis_min = 1
                    y_axis_max = 2
                    y_axis_n = 3
                elif axes_label_y == 'z [m]':
                    y_axis_min = 0
                    y_axis_max = 1
                    y_axis_n = 3
                else:
                    raise ValueError('Unknown axes_label_x')
                                        
                extent_to_use = [x_axis_min, x_axis_max, y_axis_max, y_axis_min]
                
                # transpose the slice to plot
                slice_to_plot = np.transpose(slice_to_plot)
                imi = axs[i_row,i_component].imshow(slice_to_plot, extent=extent_to_use, cmap=heat_map_type[i_row], interpolation='quadric', vmin=to_use_min, vmax=to_use_max)
                axs[i_row,i_component].set_title(f'{sub_plot_no} {data_labels[i_row]} {component_labels[i_component]}   ', fontsize=14)
                axs[i_row,i_component].invert_yaxis()
                
                # add labels
                axs[i_row,i_component].set_xlabel(axes_label_x, fontsize=14)
                axs[i_row,i_component].set_ylabel(axes_label_y, fontsize=14)

                axs[i_row,i_component].set_xticks(np.linspace(x_axis_min, x_axis_max, x_axis_n, endpoint=True))
                axs[i_row,i_component].set_xticklabels(np.round(np.linspace(x_axis_min, x_axis_max, x_axis_n, endpoint=True),1), fontsize=14)
                axs[i_row,i_component].set_yticks(np.linspace(y_axis_max, y_axis_min, y_axis_n, endpoint=True))
                axs[i_row,i_component].set_yticklabels(np.round(np.linspace(y_axis_max, y_axis_min, y_axis_n, endpoint=True),1), fontsize=14)
                                
                # stretch the aspect ratio
                axs[i_row,i_component].set_aspect('auto')
                cbar = fig.colorbar(imi, ax=axs[i_row, i_component])
                # Set colorbar to scientific notation
                cbar.formatter = ScalarFormatter(useMathText=True)
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((0, 0))
                # set font size for colorbar
                cbar.ax.yaxis.get_offset_text().set_fontsize(14)
                cbar.update_ticks()
                
        # add title for the whole figure
        title_line1 = f'{trial_save_number}' 
        title_line2 = f'{description}' 
        title_line3 = f'load case: {history_index} \n slice: {slice_label} {slice_details}'
        fig.suptitle(title_line1)
        if n_components == 1:
            # make suptitle in two lines
            fig.suptitle(f'{title_line1} \n {title_line2} \n {title_line3}' ,  fontsize=12)
        else:
            fig.suptitle(f'{title_line1} | {title_line2} | {title_line3}' ,  fontsize=12)
            
        plt.tight_layout()  # This automatically adjusts the spacing
        
        # save the plot
        save_index_append_zeros = str(save_index).zfill(2)
        # save with high resolution
        save_name = f'b-slice-{description}_hist_{history_index}_no_{save_index_append_zeros}_{slice_label}.png'
        plt.savefig(f'./{plots_save_folder}/{save_name}', dpi=300)
        print(f'Image saved --> {save_name}')
        plt.close('all')    
   
    
def get_pyvista_grid_from_data(values_list, coordinates_list, grid_to_list_map):
    
    n_x = grid_to_list_map.shape[0]
    n_y = grid_to_list_map.shape[1]
    
    data_array = np.zeros
    
    data_array = values_list.squeeze()
    
    # clone the coordinates_list and add zeros for the z axis
    points_coords_shape = coordinates_list.shape[:-1]+(3,)
    points_coords = np.zeros(points_coords_shape)
    points_coords[..., :2] = coordinates_list
 
    
    # check if data_array has the same length as points_coords
    n_points_coords = points_coords.shape[0]
    n_points_data = data_array.shape[0]
    if n_points_coords != n_points_data:
        raise ValueError(f'Error: at 2d plot n_points_coords={n_points_coords} != n_points_data={n_points_data}')
    
    cells_list = []
    cell_types_list = []

    for i_x in range(n_x-1):
        for i_y in range(n_y-1):
            i_x_0 = i_x
            i_x_1 = i_x + 1
          
            i_y_0 = i_y
            i_y_1 = i_y + 1
            indices_array = np.array([
                grid_to_list_map[i_x_0, i_y_0],
                grid_to_list_map[i_x_0, i_y_1],
                grid_to_list_map[i_x_1, i_y_1],
                grid_to_list_map[i_x_1, i_y_0],
                ])

            # check if any index is negative
            if np.any(indices_array < 0):
                continue
            
            # get mean x and y coordinates
            x_mean = (coordinates_list[indices_array[0],0] + coordinates_list[indices_array[2],0]) / 2
            y_mean = (coordinates_list[indices_array[0],1] + coordinates_list[indices_array[2],1]) / 2
            x_bounds = [44, 45]
            y_bounds = [20, 40]
            # check if the mean coordinates are in the bounds
            if (x_bounds[0] < x_mean < x_bounds[1] and y_bounds[0] < y_mean < y_bounds[1]):
                continue
            
            cell = np.array([4, *indices_array])
            cells_list.append(cell)
            cell_types_list.append(9)
                
    cells = np.vstack(cells_list)
    cell_types = np.array(cell_types_list)            

    grid = pv.UnstructuredGrid(cells, cell_types, points_coords)
    grid.point_data['scalars'] = data_array

    return grid    

def plot_true_vs_prediction_2D(true_output, predictions, coordinates_list, grid_to_list_map, history_index, description, trial_save_number, plots_save_folder):
    # know the dimensions
    n_components = true_output.shape[-1]

    component_labels = [' Pressure  ', 'Displacement x', 'Displacement y']
    sub_plot_no_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)']

    time_indices_to_plot = [29,59]
    n_time_indices_to_plot = len(time_indices_to_plot)

    ii_save = 0
    for selected_use_slice_option in [False]:
        for i_time in time_indices_to_plot:
            relative_t_index = i_time + 1
            ii_save += 1
            true_solid= true_output[i_time]
            pred_solid = predictions[i_time]

            data_labels = ['   True   ', 'Predicted', ' |Error| ', '|Error/True|']
            zero_tolerance = 1e-5 * np.max(np.abs(true_solid))
            data_array = [true_solid, pred_solid, np.abs(true_solid - pred_solid), np.abs(true_solid - pred_solid) / (np.abs(true_solid)+zero_tolerance)]
            heat_map_type = ['jet', 'jet', 'jet', 'jet']



            for n_rows in [ 3 , 4 ]:
                # Make subplots for each component (columns will represent components)
                # 3 rows will represent true, predictions and difference
    
                # Create a new figure
                if n_components == 1:
                    fig_width = int(4 * 150)
                else:
                    fig_width = int(4*3 * 150)
    
                fig_height = int(15 * 150)    
    
                fig_height_per_row = fig_height / n_rows
                fig_width_per_component = fig_width / n_components
            
                p = pv.Plotter(off_screen=True,shape=(n_rows, n_components), window_size=[fig_width, fig_height], border=False)

                title_line1 = f'{trial_save_number}' 
                title_line2 = f'{description}' 
                title_line3 = f'load case: {history_index} | time: {relative_t_index} slice: {selected_use_slice_option}'
                p.add_text(f'{title_line1}  {title_line2} {title_line3}', font_size=10, color='black', position='upper_edge')

                for i_component in range(n_components):
                    for i_row in range(n_rows):
                        # i_subplot = i_component * n_rows + i_row
                        i_subplot = i_row * n_components + i_component
                        sub_plot_no = sub_plot_no_list[i_subplot]
                        solid_to_plot = data_array[i_row][...,i_component]

                        # determine max and min values for colorbar
                        current_max = np.max(solid_to_plot)
                        current_min = np.min(solid_to_plot)
                        if i_row == 0:
                            true_max = current_max
                            true_min = current_min
                            true_max_abs = np.max(np.abs(solid_to_plot))

                        if i_row == 1:
                            to_use_max = true_max
                            to_use_min = true_min
                        elif n_rows == 4 and i_row == 2:
                            to_use_max = current_max # true_max_abs * 0.05
                            to_use_min = 0
                        elif n_rows == 4 and i_row == 3:
                            to_use_max = 0.05
                            to_use_min = 0
                        else:
                            to_use_max = current_max
                            to_use_min = current_min    



                        grid = get_pyvista_grid_from_data(solid_to_plot, coordinates_list, grid_to_list_map)
                        use_slices = selected_use_slice_option
                        if use_slices:
                            grid_to_show = grid.slice_orthogonal()
                        else:
                            grid_to_show = grid

                        p.subplot(i_row, i_component)
                        text_to_add = f'{sub_plot_no} {data_labels[i_row]} {component_labels[i_component]}'
                        if n_rows == 3:
                            p.add_text(sub_plot_no, font_size=20, position=(fig_width_per_component * 0.46,fig_height_per_row * 0.75))
                        else:
                            p.add_text(sub_plot_no, font_size=20, position=(fig_width_per_component * 0.46,fig_height_per_row * 0.77))
                            
                        if i_component == 0:
                            p.add_text(data_labels[i_row], font_size=20, position=(fig_width_per_component * 0.07,fig_height_per_row * 0.40), orientation=90)
                        if i_row == 0:
                            p.add_text(component_labels[i_component], font_size=20, position=(fig_width_per_component * 0.25,fig_height_per_row * 0.85))
                        # Add the grid to the plotter
                        scalar_bar_args = dict(
                            vertical=False,
                            title_font_size=1,
                            label_font_size=40,
                            shadow=True,
                            n_labels=3,
                            italic=True,
                            fmt="%.1e",
                            font_family="arial",
                            title= f'{sub_plot_no}',
                            position_x=0.2, 
                            position_y=0.15 if n_rows == 3 else 0.06,
                        )
                        p.add_mesh(grid_to_show, cmap='jet',lighting=True, clim=[to_use_min, to_use_max], scalar_bar_args=scalar_bar_args)


                if n_rows == 4:
                    add_name = '_rel'
                else:
                    add_name = '' 
                p.link_views()
                p.view_xy()
                # end of loop over components and rows
                save_index_append_zeros = str(ii_save).zfill(2)
                save_name = f'b-2d_{description}_hist_{history_index}_{save_index_append_zeros}_time_{relative_t_index}{add_name}.png'             
                p.screenshot(f'./{plots_save_folder}/{save_name}') 
                print(f'Image saved --> {save_name}')
                p.close()      
              
def get_max_per_component(array):
    # n_components is the last dimension
    n_components = array.shape[-1]
    # reshape to (-----, n_components)
    new_array = array.reshape(-1, n_components)
    # get max per component
    max_per_component = np.max(new_array, axis=0)
    return max_per_component

def get_l2_norm_by_l2_norm(true_output,predictions):
    # flatten the arrays
    true_output = true_output.flatten()
    predictions = predictions.flatten()
    return np.linalg.norm(true_output - predictions) / np.linalg.norm(true_output)

def get_l2_norm_by_n(array):
    return np.linalg.norm(array) / len(array.flatten())

def get_l2_norm_by_sqrt_n(array):
    return np.linalg.norm(array) / (len(array.flatten())**0.5)   

def get_l2_and_l2_of_l2_norms(true_output,predictions):
    n_histories = true_output.shape[0]
    n_time_steps = true_output.shape[1]
    n_nodes = true_output.shape[2]
    n_components = true_output.shape[3]
    
    l2_norms_per_history_time_component = np.zeros((n_histories, n_time_steps, n_components))
    for i_history in range(n_histories):
        for i_time in range(n_time_steps):
            for i_component in range(n_components):
                true_output_i = true_output[i_history,i_time,:,i_component]
                predictions_i = predictions[i_history,i_time,:,i_component]
                l2_norms_per_history_time_component[i_history, i_time, i_component] = get_l2_norm_by_l2_norm(true_output_i, predictions_i)
                
    l2_norms_per_history_component = np.zeros((n_histories, n_components))
    for i_history in range(n_histories):
        for i_component in range(n_components):
            true_output_i = true_output[i_history,:,:,i_component]
            predictions_i = predictions[i_history,:,:,i_component]
            l2_norms_per_history_component[i_history, i_component] = get_l2_norm_by_l2_norm(true_output_i, predictions_i)
            
    l2_norms_per_component = np.zeros(n_components)
    for i_component in range(n_components):
        true_output_i = true_output[:,:,:,i_component]
        predictions_i = predictions[:,:,:,i_component]
        l2_norms_per_component[i_component] = get_l2_norm_by_l2_norm(true_output_i, predictions_i)      
    
    
    l2_l2_norms_per_history_time_component_to_component = np.zeros(n_components)
    l2_l2_norms_per_history_component_to_component = np.zeros(n_components)
    for i_component in range(n_components):
        l2_l2_norms_per_history_time_component_to_component[i_component] = get_l2_norm_by_sqrt_n(l2_norms_per_history_time_component[:,:,i_component])
        l2_l2_norms_per_history_component_to_component[i_component] = get_l2_norm_by_sqrt_n(l2_norms_per_history_component[:,i_component])
        
    l2_l2_norms_per_history_time_component_to_history_component = np.zeros((n_histories, n_components))
    for i_history in range(n_histories):
        for i_component in range(n_components):
            l2_l2_norms_per_history_time_component_to_history_component[i_history,i_component] = get_l2_norm_by_sqrt_n(l2_norms_per_history_time_component[i_history,:,i_component])
    
    l2_l2_norms_per_history_time_component_to_history_component_to_component = np.zeros(n_components)
    for i_component in range(n_components):
        l2_l2_norms_per_history_time_component_to_history_component_to_component[i_component] = get_l2_norm_by_sqrt_n(l2_l2_norms_per_history_time_component_to_history_component[:,i_component])    
            
    # create a dictionary to store all the values
    l2_norms = {}
    l2_norms['l2_norms_per_history_time_component'] = l2_norms_per_history_time_component
    l2_norms['l2_norms_per_history_component'] = l2_norms_per_history_component
    l2_norms['l2_norms_per_component'] = l2_norms_per_component
    l2_norms['l2_l2_norms_per_history_time_component_to_component'] = l2_l2_norms_per_history_time_component_to_component
    l2_norms['l2_l2_norms_per_history_component_to_component'] = l2_l2_norms_per_history_component_to_component  
    l2_norms['l2_l2_norms_per_history_time_component_to_history_component'] = l2_l2_norms_per_history_time_component_to_history_component
    l2_norms['l2_l2_norms_per_history_time_component_to_history_component_to_component'] = l2_l2_norms_per_history_time_component_to_history_component_to_component
    
    return l2_norms       

def print_l2_norms_for_histories(true_output, predictions, description, save_path):
    np.set_printoptions(suppress=False, formatter={'float_kind': '{:0.3e}'.format})
    l2_norms = get_l2_and_l2_of_l2_norms(true_output, predictions)
    
    global my_logger
    
    if save_path is not None:
        # save the l2_norms dictionary to a file as an object
        to_use_path = f'{save_path}_{description}_l2_norms.pkl'
        with open(to_use_path, 'wb') as f:
            pickle.dump(l2_norms, f)
        my_logger.add_to_wait_list(f'**************************************************************************************************************')        
        my_logger.add_to_wait_list(f'L2 Norms saved to {to_use_path}')
        my_logger.add_to_wait_list(f'**************************************************************************************************************')        
    # print max values
    my_logger.add_to_wait_list(f'**************************************************************************************************************')
    my_logger.add_to_wait_list(f'Computed L2 Norms for {description}')
    my_logger.add_to_wait_list(f'**************************************************************************************************************')
    
    my_logger.add_to_wait_list(f'max l2_norms_per_history_time_component:                                   {get_max_per_component(l2_norms["l2_norms_per_history_time_component"])}')
    my_logger.add_to_wait_list(f'max l2_norms_per_history_component:                                        {get_max_per_component(l2_norms["l2_norms_per_history_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    my_logger.add_to_wait_list(f'max l2_l2_norms_per_history_time_component_to_history_component:           {get_max_per_component(l2_norms["l2_l2_norms_per_history_time_component_to_history_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    my_logger.add_to_wait_list(f'l2_norms_per_component:                                                    {(l2_norms["l2_norms_per_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    my_logger.add_to_wait_list(f'l2_l2_norms_per_history_time_component_to_component:                       {(l2_norms["l2_l2_norms_per_history_time_component_to_component"])}')
    my_logger.add_to_wait_list(f'l2_l2_norms_per_history_component_to_component:                            {(l2_norms["l2_l2_norms_per_history_component_to_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    my_logger.add_to_wait_list(f'l2_l2_norms_per_history_time_component_to_history_component_to_component:  {(l2_norms["l2_l2_norms_per_history_time_component_to_history_component_to_component"])}')

    my_logger.log_the_wait_list()
    
def use_model_and_loader_for_predictions(model, dataloader, DEVICE, trunk_input, max_batches):
    if max_batches == -1:
        max_batches = len(dataloader)
    
    max_batches = min(max_batches, len(dataloader))
        
    model.eval()  # Set the model to evaluation mode
    trunk_input = trunk_input.to(DEVICE)
    with torch.no_grad():
        all_predictions = []
        all_true_output = []
        all_load_input = []
        all_strain_input = []
        
        for i, (load_input, strain_input, true_output) in enumerate(dataloader):
            if i >= max_batches:
                break
            load_input = load_input.to(DEVICE)
            strain_input = strain_input.to(DEVICE)
            predictions = model(load_input, strain_input, trunk_input)
            all_predictions.append(predictions.cpu().numpy())
            all_true_output.append(true_output.cpu().numpy())
            all_load_input.append(load_input.cpu().numpy())
            all_strain_input.append(strain_input.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_true_output = np.concatenate(all_true_output, axis=0)
        all_load_input = np.concatenate(all_load_input, axis=0)
        all_strain_input = np.concatenate(all_strain_input, axis=0)

        global  my_logger
        my_logger.add_to_wait_list(f'-------------------------------------------------------------------')
        my_logger.add_to_wait_list(f'------------------------ New Predictions --------------------------')
        my_logger.add_to_wait_list(f'-------------------------------------------------------------------')
        my_logger.add_to_wait_list(f'all_predictions.shape: {all_predictions.shape}')
        my_logger.add_to_wait_list(f'all_true_output.shape: {all_true_output.shape}')
        my_logger.add_to_wait_list(f'all_load_input.shape: {all_load_input.shape}')
        my_logger.add_to_wait_list(f'all_strain_input.shape: {all_strain_input.shape}')
        
        my_logger.log_the_wait_list()
    
    return all_predictions, all_true_output, all_load_input, all_strain_input

def inspect_results_for_model_and_dataloader(training_data_dict, model_path, dataloader, description , mode):
    global my_logger
    
    model = training_data_dict['model']
    DEVICE = training_data_dict['DEVICE']
    trial_save_number = training_data_dict['trial_save_number']
    if mode == 'test':
        current_dataset = training_data_dict['tst_dataset']
        current_trunk_input_tensor = training_data_dict['tst_trunk_input_tensor']
        node_indices = training_data_dict['tst_node_indices']
    else:
        current_dataset = training_data_dict['my_dataset']
        current_trunk_input_tensor = training_data_dict['global_trunk_input_tensor']
        node_indices = training_data_dict['node_indices']
    
    l2_norms_save_path = None
    
    if model_path is not None:
        # load a saved model 
        load_model_checkpoint(training_data_dict, model_path)
        l2_norms_save_path = model_path.replace('.pth', '')
        
    model_predictions, model_true_output, _ , _ = use_model_and_loader_for_predictions(model, dataloader, DEVICE, current_trunk_input_tensor, max_batches=-1)
    denorm_true_output = current_dataset.denormalize_output_tensor(model_true_output)
    denorm_predictions = current_dataset.denormalize_output_tensor(model_predictions)
    
    
    my_logger.add_to_wait_list(f'normalized true output => max: {np.max(model_true_output)}, min: {np.min(model_true_output)}, mean: {np.mean(model_true_output)}, std: {np.std(model_true_output)}')
    my_logger.add_to_wait_list(f'denormalized true output => max: {np.max(denorm_true_output)}, min: {np.min(denorm_true_output)}, mean: {np.mean(denorm_true_output)}, std: {np.std(denorm_true_output)}')
    my_logger.log_and_print_the_wait_list()
    print_l2_norms_for_histories(model_true_output, model_predictions, f'normalized_{description}', None)
    print_l2_norms_for_histories(denorm_true_output, denorm_predictions, f'denormalized_{description}', l2_norms_save_path)
    
    coordinates_list = current_dataset.get_coordinates_array_list(return_as_tensor=False, normalize=False)
    true_output_list = denorm_true_output
    predictions_list = denorm_predictions
    my_logger.add_to_wait_list(f'coordinates_list.shape: {coordinates_list.shape}')
    my_logger.add_to_wait_list(f'true_output_list.shape: {true_output_list.shape}')
    my_logger.add_to_wait_list(f'predictions_list.shape: {predictions_list.shape}')    
    
    
    history_indices = [0, 1, 2]
    global plots_save_folder
    for history_index in history_indices:
        # for node_index in node_indices:
        for i_periority, node_index in enumerate(node_indices):
            node_coord = current_trunk_input_tensor[node_index,:]
            plot_true_vs_prediction(denorm_true_output, denorm_predictions, history_index, node_index, node_coord, description, trial_save_number, i_periority)

        try:
            # plot the 2D view using pyvista
            plot_true_vs_prediction_2D(true_output_list[history_index], predictions_list[history_index], coordinates_list, current_dataset.map_nodes_grid_to_list_current, history_index, description, trial_save_number, plots_save_folder)
        except Exception as e:
            my_logger.add_to_wait_list(f'Error plotting PyVista 2D view: {e}')
            my_logger.log_and_print_the_wait_list()
            print(f'Error plotting PyVista 2D view: {e}')
            
        
class MyLogger:
    def __init__(self, log_file_path, max_wait_list_length, use_csv):
        self.log_file_path = log_file_path
        self.max_wait_list_length = max_wait_list_length
        self.wait_list = []
        # open the log file and clear it
        with open(self.log_file_path, 'w') as f:
            f.write('')
        
        if use_csv:
            # remove .log from the log_file_path
            self.csv_file_path = self.log_file_path.replace('.log', '.csv')
            # open the csv file and clear it
            with open(self.csv_file_path, 'w') as f:
                f.write('')    
        
    def log(self, text):
        with open(self.log_file_path, 'a') as f:
            f.write(text + '\n')
            
    def to_csv(self, text):
        with open(self.csv_file_path, 'a') as f:
            f.write(text + '\n')        
        
    def log_list(self, text_list):
        with open(self.log_file_path, 'a') as f:
            for text in text_list:
                f.write(text + '\n')

    def log_and_print(self, text):
        print(text)
        self.log(text)
        
    def csv_and_print(self, text):
        print(text)
        self.to_csv(text)
        
    def log_and_csv_and_print(self, text):
        print(text)
        self.log(text)    
        self.to_csv(text)
        
    def log_the_wait_list(self):
        self.log_list(self.wait_list)
        self.wait_list = []    
    
    def log_and_print_the_wait_list(self):
        self.log_list(self.wait_list)
        for text in self.wait_list:
            print(text)
        self.wait_list = [] 
            
    def add_to_wait_list(self, text):
        self.wait_list.append(text)
        if len(self.wait_list) >= self.max_wait_list_length:
            self.log_the_wait_list()
            
            
            
            
            
            
                                
        