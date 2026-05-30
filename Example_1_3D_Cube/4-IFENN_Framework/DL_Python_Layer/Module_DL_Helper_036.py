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
pv.start_xvfb()
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
                # loss = torch.mean((true_data - predicted_data) ** 2) # MSELoss(reduction='mean')
                # loss = torch.sum((true_data - predicted_data) ** 2) # MSELoss(reduction='sum')
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
        self.LC_start = dl_settings['LC_start']
        self.LC_end = dl_settings['LC_end']
        self.slicing_settings = dl_settings['slicing_settings']
        self.normalization_option = dl_settings['normalization_option']
        self.load_folder_path = dl_settings['load_folder_path']
        self.output_folder_path = dl_settings['output_folder_path']
        self.load_file_name = dl_settings['load_file_name']
        self.output_file_name = dl_settings['output_file_name']
        self.coordinates_file_name = dl_settings['coordinates_file_name']
        self.max_items_saved = dl_settings['max_items_saved']
        self.read_all_the_file_once = dl_settings['read_all_the_file_once']
        self.data_info = []
        self.data_per_group = {}
        self.original_load_grid_shape = None
        self.original_strain_grid_shape = None
        self.original_output_grid_shape = None
       
        
        self.load_stats = read_pickle_data(f'{self.load_folder_path}/{self.load_file_name}stats.pkl')
        self.strain_stats = read_pickle_data(f'{self.strain_folder_path}/{self.strain_file_name}stats_var0.pkl')
        self.var0_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var0.pkl')
        self.var1_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var1.pkl')
        self.var2_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var2.pkl')
        self.var3_stats = read_pickle_data(f'{self.output_folder_path}/{self.output_file_name}stats_var3.pkl')
        
        self.original_coordinates_array_grid = h5_read_as_np(f'{self.output_folder_path}/{self.coordinates_file_name}.h5')
        
        for dataset_index, global_index in enumerate(range(self.LC_start, self.LC_end)):
            group_index, group_start, group_end, local_index = get_group_start_end_indices(global_index, 100)
            load_file_path = f'{self.load_folder_path}/{self.load_file_name}{group_start}-{group_end}.h5'
            strain_file_path = f'{self.strain_folder_path}/{self.strain_file_name}{group_start}-{group_end}.h5'
            output_file_path = f'{self.output_folder_path}/{self.output_file_name}{group_start}-{group_end}.h5'
            
            self.data_info.append((load_file_path, strain_file_path, output_file_path, local_index, dataset_index))
            
            if group_index not in self.data_per_group:
                self.data_per_group[group_index] = []
                
            self.data_per_group[group_index].append((load_file_path, strain_file_path, output_file_path, local_index, dataset_index))  
            
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
        
        global train_for_theta_only
        
        if train_for_theta_only == True:
            # only train for theta
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
        
        if self.slicing_settings['point_load_flatten']:
            load_array = load_array.reshape(load_array.shape[0], -1)
            
        if self.slicing_settings['point_strain_flatten']:
            strain_array = strain_array.reshape(strain_array.shape[0], -1)    
            
        if self.slicing_settings['point_output_flatten']:
            output_array = output_array.reshape(output_array.shape[0], -1, output_array.shape[-1])

        load_array = normalize_data(load_array, self.load_stats, self.normalization_option)

        strain_array = normalize_data(strain_array, self.strain_stats, self.normalization_option)

        output_array[..., 0] = normalize_data(output_array[..., 0], self.var0_stats, self.normalization_option)
        if train_for_theta_only == False:
            output_array[..., 1] = normalize_data(output_array[..., 1], self.var1_stats, self.normalization_option)
            output_array[..., 2] = normalize_data(output_array[..., 2], self.var2_stats, self.normalization_option)
            output_array[..., 3] = normalize_data(output_array[..., 3], self.var3_stats, self.normalization_option)
        
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
    
    def denormalize_output_tensor(self, output_tensor):
        global train_for_theta_only
        # check if numpy or tensor
        if isinstance(output_tensor, torch.Tensor):
            denormalized_output_tensor = torch.zeros_like(output_tensor)
        else:
            denormalized_output_tensor = np.zeros_like(output_tensor)    
        
        denormalized_output_tensor[..., 0] = denormalize_data(output_tensor[..., 0], self.var0_stats, self.normalization_option)
        if train_for_theta_only == False:
            denormalized_output_tensor[..., 1] = denormalize_data(output_tensor[..., 1], self.var1_stats, self.normalization_option)
            denormalized_output_tensor[..., 2] = denormalize_data(output_tensor[..., 2], self.var2_stats, self.normalization_option)
            denormalized_output_tensor[..., 3] = denormalize_data(output_tensor[..., 3], self.var3_stats, self.normalization_option)
        
        return denormalized_output_tensor
    
    def normalize_output_tensor(self, output_tensor):
        global train_for_theta_only
        # check if numpy or tensor
        if isinstance(output_tensor, torch.Tensor):
            normalized_output_tensor = torch.zeros_like(output_tensor, dtype=torch.float32)
        else:
            normalized_output_tensor = np.zeros_like(output_tensor)    
        
        normalized_output_tensor[..., 0] = normalize_data(output_tensor[..., 0], self.var0_stats, self.normalization_option)
        if train_for_theta_only == False:
            normalized_output_tensor[..., 1] = normalize_data(output_tensor[..., 1], self.var1_stats, self.normalization_option)
            normalized_output_tensor[..., 2] = normalize_data(output_tensor[..., 2], self.var2_stats, self.normalization_option)
            normalized_output_tensor[..., 3] = normalize_data(output_tensor[..., 3], self.var3_stats, self.normalization_option)
        
        return normalized_output_tensor
     
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
        
        load_start = self.slicing_settings['point_load_start']
        load_skip = self.slicing_settings['point_load_skip']
        load_end = self.slicing_settings['point_load_end']
        strain_start = self.slicing_settings['point_strain_start']
        strain_skip = self.slicing_settings['point_strain_skip']
        strain_end = self.slicing_settings['point_strain_end']
        output_start = self.slicing_settings['point_output_start']
        output_skip = self.slicing_settings['point_output_skip']
        output_end = self.slicing_settings['point_output_end']
        load_time_start = self.slicing_settings['load_time_start']
        load_time_skip = self.slicing_settings['load_time_skip']
        load_time_end = self.slicing_settings['load_time_end']
        strain_time_start = self.slicing_settings['strain_time_start']
        strain_time_skip = self.slicing_settings['strain_time_skip']
        strain_time_end = self.slicing_settings['strain_time_end']
        
        # loop over keys in data_per_group
        for group_index in self.data_per_group:
            required_data_in_group = self.data_per_group[group_index]
            load_file_path, strain_file_path , output_file_path, _ , _ = required_data_in_group[0]
            
            
            with h5py.File(load_file_path, 'r') as load_file , h5py.File(output_file_path, 'r') as output_file, h5py.File(strain_file_path, 'r') as strain_file:
                if self.read_all_the_file_once:
                    my_logger.log_and_print(f'Loading all at once at group {group_index}')
                    load_data_all = np.array(load_file['data'])
                    strain_data_all = np.array(strain_file['data'])
                    output_data_all = np.array(output_file['data'])
                    for _,_,_, local_index, dataset_index in required_data_in_group:
                        load_data = load_data_all[local_index, load_time_start:load_time_end:load_time_skip, load_start[0]:load_end[0]:load_skip[0], load_start[1]:load_end[1]:load_skip[1], load_start[2]:load_end[2]:load_skip[2]]
                        strain_data = strain_data_all[local_index, strain_time_start:strain_time_end:strain_time_skip, strain_start[0]:strain_end[0]:strain_skip[0], strain_start[1]:strain_end[1]:strain_skip[1], strain_start[2]:strain_end[2]:strain_skip[2], :]
                        output_data = output_data_all[local_index, load_time_start:load_time_end:load_time_skip, output_start[0]:output_end[0]:output_skip[0], output_start[1]:output_end[1]:output_skip[1], output_start[2]:output_end[2]:output_skip[2], :]
                        load_array, strain_array, output_array = self.post_process_loaded_data(load_data, strain_data, output_data)
                        current_cashed_items_dict[dataset_index] = (load_array, strain_array, output_array)
                    
                else:    
                    my_logger.log_and_print(f'Loading index by index at group {group_index}')
                    for _,_,_, local_index, dataset_index in required_data_in_group:
                        load_data = load_file['data'][local_index, load_time_start:load_time_end:load_time_skip, load_start[0]:load_end[0]:load_skip[0], load_start[1]:load_end[1]:load_skip[1], load_start[2]:load_end[2]:load_skip[2]]
                        strain_data = strain_file['data'][local_index, strain_time_start:strain_time_end:strain_time_skip, strain_start[0]:strain_end[0]:strain_skip[0], strain_start[1]:strain_end[1]:strain_skip[1], strain_start[2]:strain_end[2]:strain_skip[2], :]
                        output_data = output_file['data'][local_index, load_time_start:load_time_end:load_time_skip, output_start[0]:output_end[0]:output_skip[0], output_start[1]:output_end[1]:output_skip[1], output_start[2]:output_end[2]:output_skip[2], :]
                        load_array, strain_array, output_array = self.post_process_loaded_data(load_data, strain_data, output_data)
                        current_cashed_items_dict[dataset_index] = (load_array, strain_array, output_array)
        
        assert len(current_cashed_items_dict) == len(self.data_info), 'Error : All data not loaded correctly in pre_load_all_data'
        
        end_time = time.time()            
        my_logger.log_and_print(f'..... Preloading all data ..... ended at {time.ctime()} and took {(end_time - start_time)/60} minutes')    
                
        
    def get_new_item(self, idx):
        global main_cashed_items_dict
        global test_cashed_items_dict
        
        
        load_file_path, strain_file_path, output_file_path, local_index, dataset_index = self.data_info[idx]
        
        assert dataset_index == idx , 'Error : dataset_index and idx should be the same'
        
        load_start = self.slicing_settings['point_load_start']
        load_skip = self.slicing_settings['point_load_skip']
        load_end = self.slicing_settings['point_load_end']
        strain_start = self.slicing_settings['point_strain_start']
        strain_skip = self.slicing_settings['point_strain_skip']
        strain_end = self.slicing_settings['point_strain_end']
        output_start = self.slicing_settings['point_output_start']
        output_skip = self.slicing_settings['point_output_skip']
        output_end = self.slicing_settings['point_output_end']
        load_time_start = self.slicing_settings['load_time_start']
        load_time_skip = self.slicing_settings['load_time_skip']
        load_time_end = self.slicing_settings['load_time_end']
        strain_time_start = self.slicing_settings['strain_time_start']
        strain_time_skip = self.slicing_settings['strain_time_skip']
        strain_time_end = self.slicing_settings['strain_time_end']
        
        with h5py.File(load_file_path, 'r') as h5_file:
            load_data = h5_file['data'][local_index, load_time_start:load_time_end:load_time_skip, load_start[0]:load_end[0]:load_skip[0], load_start[1]:load_end[1]:load_skip[1], load_start[2]:load_end[2]:load_skip[2]]
        with h5py.File(strain_file_path, 'r') as h5_file:
            strain_data = h5_file['data'][local_index, strain_time_start:strain_time_end:strain_time_skip, strain_start[0]:strain_end[0]:strain_skip[0], strain_start[1]:strain_end[1]:strain_skip[1], strain_start[2]:strain_end[2]:strain_skip[2], :]
        with h5py.File(output_file_path, 'r') as h5_file:
            output_data = h5_file['data'][local_index, load_time_start:load_time_end:load_time_skip, output_start[0]:output_end[0]:output_skip[0], output_start[1]:output_end[1]:output_skip[1], output_start[2]:output_end[2]:output_skip[2], :]
        
        load_array, strain_array, output_array = self.post_process_loaded_data(load_data, strain_data, output_data)
        
        if self.mode == 'test':
            current_cashed_items_dict = test_cashed_items_dict
        else:
            current_cashed_items_dict = main_cashed_items_dict
            
        if len(current_cashed_items_dict) < self.max_items_saved:
            current_cashed_items_dict[idx] = (load_array, strain_array, output_array)
        
        return load_array, strain_array, output_array
    
    
    def get_coordinates_array(self, return_as_tensor):
        output_start = self.slicing_settings['point_output_start']
        output_skip = self.slicing_settings['point_output_skip']
        output_end = self.slicing_settings['point_output_end']
        sliced_coordinates_array = self.original_coordinates_array_grid[output_start[0]:output_end[0]:output_skip[0], output_start[1]:output_end[1]:output_skip[1], output_start[2]:output_end[2]:output_skip[2], :]
        if self.slicing_settings['point_output_flatten']:
            sliced_coordinates_array = sliced_coordinates_array.reshape(-1, sliced_coordinates_array.shape[-1])
        if return_as_tensor:
            return torch.tensor(sliced_coordinates_array, dtype=torch.float32)
        else:
            return sliced_coordinates_array
    
    def get_index_of_coordinates_list(self, coordinates_list):
        sliced_coordinates_array = self.get_coordinates_array(return_as_tensor=False)
        index_list = []
        for coordinates in coordinates_list:
            # find the nearest point in the sliced_coordinates_array_per_id
            index = np.argmin(np.linalg.norm(sliced_coordinates_array - coordinates, axis=1))
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

def plot_true_vs_prediction(true_output, predictions, history_index, node_index, node_coords, description, trial_save_number):

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
    labels = ['theta', 'ux', 'uy', 'uz']
    
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
    title_line3 = f'coord: ({node_coords[0]:.3f},{node_coords[1]:.3f},{node_coords[2]:.3f})'
    
    if n_components == 1:
        # make suptitle in two lines
        fig.suptitle(f'{title_line0} \n {title_line1} \n {title_line2} \n {title_line3}' ,  fontsize=12)
    else:
        fig.suptitle(f'{title_line0} | {title_line1} | {title_line2} | {title_line3}' ,  fontsize=12)
            
    plt.tight_layout()  # This automatically adjusts the spacing
    
    # save the plot
    global plots_save_folder
    plt.savefig(f'./{plots_save_folder}/c-predictions_{description}_hist_{history_index}_node_{node_index}.png')
    plt.close('all')
    
def plot_true_vs_prediction_slice(true_output, predictions, history_index, description, trial_save_number, plots_save_folder):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*changed.*")
    plt.rc('font', family= 'serif')  # 'Times New Roman'
    plt.rc('font', size=14)
    # know the dimensions
    n_time_steps = true_output.shape[0]
    x_dim = true_output.shape[1]
    y_dim = true_output.shape[2]
    z_dim = true_output.shape[3]
    n_components = true_output.shape[-1]
    
    component_labels = [r'$\theta$', r'$u_x$', r'$u_y$', r'$u_z$']
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
    for iz,z_index in enumerate([z_dim // 2]):
        for iy, y_index in enumerate([y_dim // 2]):
            si+=1
            relative_z_index = round(z_index/z_dim,1)
            relative_y_index = round(y_index/y_dim,1)
                
            true_slices.append(true_output[:, :, y_index, z_index, :])
            pred_slices.append(predictions[:, :, y_index, z_index, :])
            slices_details.append(f'at y={relative_y_index}, z={relative_z_index}')
            slices_labels.append(f't-x')
            axes_labels.append(['time [s]', 'x [m]'])
            save_indices.append(si)
    
    # t-y slices (line in space)
    for iz,z_index in enumerate([z_dim // 2]):
        for ix, x_index in enumerate([x_dim // 2]):
            si+=1
            relative_z_index = round(z_index/z_dim,1)
            relative_x_index = round(x_index/x_dim,1)
                            
            true_slices.append(true_output[:, x_index, :, z_index, :])
            pred_slices.append(predictions[:, x_index, :, z_index, :])
            slices_details.append(f'at x={relative_x_index}, z={relative_z_index}')
            slices_labels.append(f't-y')
            axes_labels.append(['time [s]', 'y [m]'])
            save_indices.append(si)
    
    # t-z slices (line in space)
    for iy, y_index in enumerate([y_dim // 2]):
        for ix, x_index in enumerate([x_dim // 2]):
            si+=1
            relative_y_index = round(y_index/y_dim,1)
            relative_x_index = round(x_index/x_dim,1)
                            
            true_slices.append(true_output[:, x_index, y_index, :, :])
            pred_slices.append(predictions[:, x_index, y_index, :, :])
            slices_details.append(f'at x={relative_x_index}, y={relative_y_index}')
            slices_labels.append(f't-z')
            axes_labels.append(['time [s]', 'z [m]'])
            save_indices.append(si)

    # x-y slices
    for it,t_index in enumerate([9,49,74,99]):
        for iz,z_index in enumerate([z_dim // 2]):
            si+=1
            relative_z_index = round(z_index/z_dim,1)
            relative_t_index = t_index + 1
            
            true_slices.append(true_output[t_index, :, :, z_index, :])
            pred_slices.append(predictions[t_index, :, :, z_index, :])
            slices_details.append(f'at z={relative_z_index}, t={relative_t_index}')
            slices_labels.append(f'x-y')
            axes_labels.append(['x [m]', 'y [m]'])
            save_indices.append(si)
    
    # x-z slices
    for it,t_index in enumerate([9,49,74,99]):
        for iy, y_index in enumerate([y_dim // 2]):
            si+=1
            relative_y_index = round(y_index/y_dim,1)
            relative_t_index = t_index + 1
            
            true_slices.append(true_output[t_index, :, y_index, :, :])
            pred_slices.append(predictions[t_index, :, y_index, :, :])
            slices_details.append(f'at y={relative_y_index}, t={relative_t_index}')
            slices_labels.append(f'x-z')
            axes_labels.append(['x [m]', 'z [m]'])
            save_indices.append(si)      
            
    # y-z slices
    for it,t_index in enumerate([9,49,74,99]):
        for ix, x_index in enumerate([x_dim // 2]):
            si+=1
            relative_x_index = round(x_index/x_dim,1)
            relative_t_index = t_index + 1
            
            true_slices.append(true_output[t_index, x_index, :, :, :])
            pred_slices.append(predictions[t_index, x_index, :, :, :])
            slices_details.append(f'at x={relative_x_index}, t={relative_t_index}')
            slices_labels.append(f'y-z')
            axes_labels.append(['y [m]', 'z [m]'])
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
                    x_axis_max = 18000
                    x_axis_n = 3
                else:
                    x_axis_min = 0
                    x_axis_max = 1
                    x_axis_n = 6
                
                if axes_label_y == 'time [s]':
                    y_axis_min = 0
                    y_axis_max = 18000
                    y_axis_n = 3
                else:
                    y_axis_min = 0
                    y_axis_max = 1
                    y_axis_n = 6
                    
                x_axis_n_true = slice_to_plot.shape[1]    
                y_axis_n_true = slice_to_plot.shape[0]
                
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
                # format the colorbar (ensure that the colorbar is not in scientific notation)
                # # delete existing colorbar
                # axs[i_row,i_component].images[-1].colorbar.remove()
                # add colorbar for each imshow
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
        title_line3 = f'load case: {history_index} | slice: {slice_label} {slice_details}'
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
        save_name = f'b-slice-{slice_label}_{description}_hist_{history_index}_slice_{save_index_append_zeros}.png'
        plt.savefig(f'./{plots_save_folder}/{save_name}', dpi=300)
        print(f'Image saved --> {save_name}')
        plt.close('all')    
   
    
def get_pyvista_grid_from_data(data_array, refinement_level):
    zoom_factor = np.array([1, 1, 1]) * refinement_level
    if refinement_level > 0:
        # zoom the data array
        fine_data_array = zoom(data_array, zoom_factor, order=2)
        print( f"zoom_factor = {zoom_factor} applied to data_array.shape = {data_array.shape} -> fine_data_array.shape = {fine_data_array.shape}")
    else:
        fine_data_array = data_array
    # reorder the axes to match the PyVista convention 012 -> 201 (use transpose)
    fine_data_array = np.transpose(fine_data_array, (2, 0, 1))
    # Create the spatial reference
    grid = pv.ImageData()
    # CELL data
    grid.dimensions = np.array(fine_data_array.shape) # + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid['scalars'] = fine_data_array.flatten(order="F")
    
    return grid    

def plot_true_vs_prediction_3D(true_output, predictions, history_index, description, trial_save_number, plots_save_folder):
    # know the dimensions
    n_time_steps = true_output.shape[0]
    x_dim = true_output.shape[1]
    y_dim = true_output.shape[2]
    z_dim = true_output.shape[3]
    n_components = true_output.shape[-1]

    component_labels = ['Temp', ' Ux ', ' Uy ', ' Uz ']
    sub_plot_no_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)']

    time_indices_to_plot = [9,24,49,74,85,99]
    n_time_indices_to_plot = len(time_indices_to_plot)

    ii_save = 0
    all_components_in_one_row = True
    for selected_use_slice_option in [False]:
        for i_time in time_indices_to_plot:
            relative_t_index = i_time + 1
            ii_save += 1
            true_solid= true_output[i_time]
            pred_solid = predictions[i_time]

            abs_error = np.abs(true_solid - pred_solid)
            rel_error = np.zeros(true_solid.shape)
            max_rel_cap = 0.05 # np.max(solid_to_plot)
            for i_component in range(n_components):
                max_abs_true = np.max(np.abs(true_solid[...,i_component]))
                div_tol = max_abs_true * 1e-5
                rel_error[...,i_component] = np.abs(true_solid[...,i_component] - pred_solid[...,i_component]) / (np.abs(true_solid[...,i_component]) + div_tol)
            data_labels_org = ['True', 'Pred', 'Abs.', 'Rel.']
            data_array_org = [true_solid, pred_solid, abs_error, rel_error]
            
            plot_options = ['abs', 'rel', 'abs+rel']
            for plot_option in plot_options:
                if plot_option == 'abs':
                    data_types = 3
                    select_indices = [0, 1, 2]
                    data_labels = [data_labels_org[i] for i in select_indices]
                    data_array = [data_array_org[i] for i in select_indices]
                elif plot_option == 'rel':
                    data_types = 3
                    select_indices = [0, 1, 3]
                    data_labels = [data_labels_org[i] for i in select_indices]
                    data_array = [data_array_org[i] for i in select_indices]
                elif plot_option == 'abs+rel':
                    data_types = 4
                    data_labels = data_labels_org
                    data_array = data_array_org
                

                # Make subplots for each component (columns will represent components)
                # 3 rows will represent true, predictions and difference
                if all_components_in_one_row:
                    n_rows = data_types
                    n_cols = n_components
                else:
                    n_rows = n_components
                    n_cols = data_types  

                fig_width = int(n_cols * 900)

                fig_height = int(n_rows * 600)

                fig_height_per_row = fig_height / n_rows
                fig_width_per_col = fig_width / n_cols

                p = pv.Plotter(off_screen=True,shape=(n_rows, n_cols), window_size=[fig_width, fig_height], border=False)

                # adjust subplots
                rel_x_inc = 1.0 / n_cols
                rel_y_inc = 1.0 / n_rows
                x_vp_overlap = 0.0 * rel_x_inc
                y_vp_overlap = 0.0 * rel_y_inc
                for i_row in range(n_rows):
                    for i_col in range(n_cols):
                        idx = i_row * n_cols + i_col
                        x_vp_min = i_col * rel_x_inc - x_vp_overlap
                        x_vp_max = (i_col + 1) * rel_x_inc + x_vp_overlap
                        y_vp_min = (n_rows - i_row - 1) * rel_y_inc - y_vp_overlap
                        y_vp_max = (n_rows - i_row) * rel_y_inc + y_vp_overlap
                        p.renderers[idx].viewport = (x_vp_min, y_vp_min, x_vp_max, y_vp_max)

                title_line1 = f'{trial_save_number}' 
                title_line2 = f'{description}' 
                title_line3 = f'load case:{history_index} time:{relative_t_index} slice:{selected_use_slice_option}'
                p.add_text(f'{title_line1}  {title_line2}  {title_line3}', font_size=10, color='black', position='upper_left', viewport=True)



                for i_row in range(n_rows):
                    for i_col in range(n_cols):
                        i_subplot = i_row * n_cols + i_col
                        sub_plot_no = sub_plot_no_list[i_subplot]
                        if all_components_in_one_row:
                            i_component = i_col
                            i_data_type = i_row
                        else:
                            i_component = i_row
                            i_data_type = i_col
                        
                        solid_to_plot = data_array[i_data_type][...,i_component]    
                        if i_data_type == 0 or i_data_type == 1:
                            to_use_max = np.max(data_array[0][...,i_component])
                            to_use_min = np.min(data_array[0][...,i_component])
                        elif i_data_type == 2 and plot_option == 'abs':
                            to_use_max = np.max(solid_to_plot)
                            to_use_min = 0
                        elif i_data_type == 2 and plot_option == 'rel':
                            to_use_max = max_rel_cap
                            to_use_min = 0
                        elif i_data_type == 2 and plot_option == 'abs+rel':
                            to_use_max = np.max(solid_to_plot)
                            to_use_min = 0
                        elif i_data_type == 3 and plot_option == 'abs+rel':
                            to_use_max = max_rel_cap
                            to_use_min = 0.0

                        grid = get_pyvista_grid_from_data(solid_to_plot, 0)
                        use_slices = selected_use_slice_option
                        if use_slices:
                            grid_to_show = grid.slice_orthogonal()
                        else:
                            grid_to_show = grid

                        p.subplot(i_row,i_col)
                        scalar_bar_is_vertical = True
                        scalar_bar_pos_x = 0.62
                        scalar_bar_pos_y = 0.13
                        scalar_bar_width = 0.08
                        scalar_bar_height = 0.74
                        scalar_bar_args = dict(
                            vertical=scalar_bar_is_vertical,
                            title_font_size=1,
                            label_font_size=74,
                            shadow=True,
                            n_labels=3,
                            italic=False,
                            fmt="%.1e",
                            font_family="times", # arial, courier, times
                            # add latex title (ux, uy, uz, theta)
                            title= f'{sub_plot_no}',
                            position_x=scalar_bar_pos_x, 
                            position_y=scalar_bar_pos_y,
                            width=scalar_bar_width,
                            height=scalar_bar_height,
                            # below_label='test',
                        )
                        p.add_mesh(grid_to_show, cmap='jet',lighting=True, clim=[to_use_min, to_use_max], scalar_bar_args=scalar_bar_args)
                        
                if all_components_in_one_row:
                    add_name = f'r_{plot_option}'
                else:
                    add_name = f'c_{plot_option}'
                p.link_views()
                p.view_isometric()
                p.camera.zoom(0.95)
                # Access the camera
                camera = p.camera

                # Get position and focal point as numpy arrays
                cam_pos = np.array(camera.GetPosition())
                cam_focal = np.array(camera.GetFocalPoint())
                cam_up = np.array(camera.GetViewUp())
                # Compute view direction (normalized)
                view_dir = cam_focal - cam_pos
                view_dir /= np.linalg.norm(view_dir)
                # Compute right vector (normalized)
                cam_right = np.cross(view_dir, cam_up)
                cam_right /= np.linalg.norm(cam_right)
                # Move camera position and focal point along right vector
                cam_offset = 13  # adjust how far you want to move to the right
                focal_offset = 14  # adjust how far you want to move the focal point
                new_cam_pos = cam_pos + cam_offset * cam_right
                new_cam_focal = cam_focal + focal_offset * cam_right
                # Apply new camera position
                camera.SetPosition(new_cam_pos.tolist())
                camera.SetFocalPoint(new_cam_focal.tolist())
                camera.SetViewUp(cam_up.tolist())
                p.subplot(0,0)
                p.add_axes(viewport=(0.0, 0.8, 0.15, 0.98),xlabel='Z', ylabel='X', zlabel='Y', color='black')


                # end of loop over components and rows
                save_index_append_zeros = str(ii_save).zfill(2)
                save_name = f'b-3d_{description}_hist_{history_index}_{save_index_append_zeros}_time_{relative_t_index}_{add_name}.png'             
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
    
    # my_logger.add_to_wait_list(f'max l2_norms_per_history_node_component:                                   {get_max_per_component(l2_norms["l2_norms_per_history_node_component"])}')
    my_logger.add_to_wait_list(f'max l2_norms_per_history_time_component:                                   {get_max_per_component(l2_norms["l2_norms_per_history_time_component"])}')
    my_logger.add_to_wait_list(f'max l2_norms_per_history_component:                                        {get_max_per_component(l2_norms["l2_norms_per_history_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    # my_logger.add_to_wait_list(f'max l2_l2_norms_per_history_node_component_to_history_component:           {get_max_per_component(l2_norms["l2_l2_norms_per_history_node_component_to_history_component"])}')
    my_logger.add_to_wait_list(f'max l2_l2_norms_per_history_time_component_to_history_component:           {get_max_per_component(l2_norms["l2_l2_norms_per_history_time_component_to_history_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    my_logger.add_to_wait_list(f'l2_norms_per_component:                                                    {(l2_norms["l2_norms_per_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    # my_logger.add_to_wait_list(f'l2_l2_norms_per_history_node_component_to_component:                       {(l2_norms["l2_l2_norms_per_history_node_component_to_component"])}')
    my_logger.add_to_wait_list(f'l2_l2_norms_per_history_time_component_to_component:                       {(l2_norms["l2_l2_norms_per_history_time_component_to_component"])}')
    my_logger.add_to_wait_list(f'l2_l2_norms_per_history_component_to_component:                            {(l2_norms["l2_l2_norms_per_history_component_to_component"])}')
    
    my_logger.add_to_wait_list(f'--------------------------------------------------------------------------------------------------------------')
    
    # my_logger.add_to_wait_list(f'l2_l2_norms_per_history_node_component_to_history_component_to_component:  {(l2_norms["l2_l2_norms_per_history_node_component_to_history_component_to_component"])}')
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
    # model_predictions, model_true_output, model_input_loads, model_input_strains = use_model_and_loader_for_predictions(model, dataloader, DEVICE, current_trunk_input_tensor, max_batches=-1)
    denorm_true_output = current_dataset.denormalize_output_tensor(model_true_output)
    denorm_predictions = current_dataset.denormalize_output_tensor(model_predictions)
    
    unflattened_true_output = current_dataset.unflatten_tensor(denorm_true_output, current_dataset.original_output_grid_shape, batch_included=True)
    unflattened_predictions = current_dataset.unflatten_tensor(denorm_predictions, current_dataset.original_output_grid_shape, batch_included=True)
    
    my_logger.add_to_wait_list(f'normalized true output => max: {np.max(model_true_output)}, min: {np.min(model_true_output)}, mean: {np.mean(model_true_output)}, std: {np.std(model_true_output)}')
    my_logger.add_to_wait_list(f'denormalized true output => max: {np.max(denorm_true_output)}, min: {np.min(denorm_true_output)}, mean: {np.mean(denorm_true_output)}, std: {np.std(denorm_true_output)}')
    my_logger.log_and_print_the_wait_list()
    print_l2_norms_for_histories(model_true_output, model_predictions, f'normalized_{description}', None)
    print_l2_norms_for_histories(denorm_true_output, denorm_predictions, f'denormalized_{description}', l2_norms_save_path)
    
    history_indices = [0, 1, 2]
    global plots_save_folder
    for history_index in history_indices:
        for node_index in node_indices:
            node_coord = current_trunk_input_tensor[node_index,:]
            plot_true_vs_prediction(denorm_true_output, denorm_predictions, history_index, node_index, node_coord, description, trial_save_number)

        my_logger.add_to_wait_list(f'SKIP:: plotting 3D view')
        my_logger.log_and_print_the_wait_list()
        plot_true_vs_prediction_slice(unflattened_true_output[history_index], unflattened_predictions[history_index], history_index, description, trial_save_number, plots_save_folder)
        
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
            
            
            
            
            
            
                                
        