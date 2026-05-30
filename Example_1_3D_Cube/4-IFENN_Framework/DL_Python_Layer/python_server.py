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
import Module_DL_Helper_036 as MDLH
import Module_Solution_Loader_Helper_006 as MSLH
import mmap
import posix_ipc
import struct
import time
import signal
import sys

def load_shared_memories():
    # Shared memory and semaphore names
    st_shared_mem_name0 = "/shared_mem0"
    st_shared_mem_name1 = "/shared_mem1"
    st_shared_mem_name2 = "/shared_mem2"
    st_shared_mem_name3 = "/shared_mem3"

    st_semaphore_python_ready = "/semaphore_python_ready"
    st_semaphore_cpp_ready = "/semaphore_cpp_ready"
    st_semaphore_solution_saved = "/semaphore_solution_saved"

    # Define the shared memory structure format
    shared_flags_format = "ff"  # can be written as "i" but python will consider i as f
    shared_input_format = "512f"  # n floats 
    shared_output_format = "29791f"  # n floats
    shared_check_format = "512f"  # n floats
    
    
    # Compute the size of the shared memory
    mem_size_flags = struct.calcsize(shared_flags_format)
    mem_size_input = struct.calcsize(shared_input_format)
    mem_size_output = struct.calcsize(shared_output_format)
    mem_size_check = struct.calcsize(shared_check_format)

    # Open shared memory and semaphores
    # Open or create shared memory and semaphores
    try:
        shm_fd0 = posix_ipc.SharedMemory(st_shared_mem_name0)
    except (posix_ipc.ExistentialError, ValueError):
        shm_fd0 = posix_ipc.SharedMemory(st_shared_mem_name0, posix_ipc.O_CREX, size=mem_size_flags)
    
    try:
        shm_fd1 = posix_ipc.SharedMemory(st_shared_mem_name1)
    except (posix_ipc.ExistentialError, ValueError):
        shm_fd1 = posix_ipc.SharedMemory(st_shared_mem_name1, posix_ipc.O_CREX, size=mem_size_input)

    try:
        shm_fd2 = posix_ipc.SharedMemory(st_shared_mem_name2)
    except (posix_ipc.ExistentialError, ValueError):
        shm_fd2 = posix_ipc.SharedMemory(st_shared_mem_name2, posix_ipc.O_CREX, size=mem_size_output)

    try:
        shm_fd3 = posix_ipc.SharedMemory(st_shared_mem_name3)
    except (posix_ipc.ExistentialError, ValueError):
        shm_fd3 = posix_ipc.SharedMemory(st_shared_mem_name3, posix_ipc.O_CREX, size=mem_size_check)

    try:
        sem_python_output_ready = posix_ipc.Semaphore(st_semaphore_python_ready)
    except posix_ipc.ExistentialError:
        sem_python_output_ready = posix_ipc.Semaphore(st_semaphore_python_ready, posix_ipc.O_CREX)

    try:
        sem_cpp_input_ready = posix_ipc.Semaphore(st_semaphore_cpp_ready)
    except posix_ipc.ExistentialError:
        sem_cpp_input_ready = posix_ipc.Semaphore(st_semaphore_cpp_ready, posix_ipc.O_CREX)
        
    try:
        sem_solution_saved = posix_ipc.Semaphore(st_semaphore_solution_saved)
    except posix_ipc.ExistentialError:
        sem_solution_saved = posix_ipc.Semaphore(st_semaphore_solution_saved, posix_ipc.O_CREX)    

    # Map the shared memory
    shared_flags = mmap.mmap(shm_fd0.fd, mem_size_flags, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    shared_input = mmap.mmap(shm_fd1.fd, mem_size_input, mmap.MAP_SHARED, mmap.PROT_READ)
    shared_output = mmap.mmap(shm_fd2.fd, mem_size_output, mmap.MAP_SHARED, mmap.PROT_WRITE)
    shared_check = mmap.mmap(shm_fd3.fd, mem_size_check, mmap.MAP_SHARED, mmap.PROT_READ)
    
    # Assign signal handlers
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Assign the shared memories to the global variable
    shm_object_dict["shared_flags"] = shared_flags
    shm_object_dict["shared_input"] = shared_input
    shm_object_dict["shared_output"] = shared_output
    shm_object_dict["shared_check"] = shared_check
    shm_object_dict["sem_python_output_ready"] = sem_python_output_ready
    shm_object_dict["sem_cpp_input_ready"] = sem_cpp_input_ready
    shm_object_dict["sem_solution_saved"] = sem_solution_saved
    shm_object_dict["shm_fd0"] = shm_fd0
    shm_object_dict["shm_fd1"] = shm_fd1
    shm_object_dict["shm_fd2"] = shm_fd2
    shm_object_dict["shm_fd3"] = shm_fd3
    shm_object_dict["shared_flags_format"] = shared_flags_format
    shm_object_dict["shared_input_format"] = shared_input_format
    shm_object_dict["shared_output_format"] = shared_output_format
    shm_object_dict["shared_check_format"] = shared_check_format

def are_qeual(val1, val2):
    return abs(val1 - val2) < 1e-8

def start_input_out_process():
    # Unpack the shared memories
    shared_flags = shm_object_dict["shared_flags"]
    shared_input = shm_object_dict["shared_input"]
    shared_output = shm_object_dict["shared_output"]
    shared_check = shm_object_dict["shared_check"]
    sem_python_output_ready = shm_object_dict["sem_python_output_ready"]
    sem_cpp_input_ready = shm_object_dict["sem_cpp_input_ready"]
    sem_solution_saved = shm_object_dict["sem_solution_saved"]
    shared_flags_format = shm_object_dict["shared_flags_format"]
    shared_input_format = shm_object_dict["shared_input_format"]
    shared_output_format = shm_object_dict["shared_output_format"]
    shared_check_format = shm_object_dict["shared_check_format"]
    
    while True:
        # Wait for C++ to signal input is ready or output is ready
        sem_cpp_input_ready.acquire()
        print("One python process Started ######==>")

        # Read the input data from shared memory
        shared_flags.seek(0) # Move the reading pointer to the beginning of the shared memory
        flags_data = shared_flags.read(struct.calcsize(shared_flags_format)) # Read the given number of bytes
        load_case_index, t_step_no = struct.unpack(shared_flags_format, flags_data) # Unpack the bytes into the variables
        load_case_index = int(load_case_index)
        t_step_no = int(t_step_no)
        
        shared_input.seek(0) # Move the reading pointer to the beginning of the shared memory
        input_data = shared_input.read(struct.calcsize(shared_input_format)) # Read the given number of bytes
        input_vector = struct.unpack(shared_input_format, input_data) # Unpack the bytes into the variables
        # print type of input_vector
        input_tensor = torch.tensor(input_vector).reshape(1, -1) # add time dimension to the beginning
        
        
        shared_check.seek(0)
        check_data = shared_check.read(struct.calcsize(shared_check_format))
        check_vector = struct.unpack(shared_check_format, check_data)

        
        #################################################
        # Load target data for the given load case index
        #################################################
        if ('target_load_case_index' not in p_dict) or (p_dict['target_load_case_index'] != load_case_index):
            simulation_start_time = time.time()
            update_pytorch_dict_for_load_case(load_case_index)

        my_dataset = p_dict['my_dataset']
        model = p_dict['model']
        DEVICE = p_dict['DEVICE']
        trunk_input_tensor = p_dict['trunk_input_tensor']
        load_data_i = p_dict['load_data_i']
        strain_data_i = p_dict['strain_data_i']
        output_data_i = p_dict['output_data_i']
        my_logger = p_dict['my_logger']
        #######################################################################
        my_logger.add_to_wait_list(f"Received check_vector => max: {max(check_vector)} min: {min(check_vector)}")
        my_logger.add_to_wait_list(f"Received load_case_index: {load_case_index}")
        my_logger.add_to_wait_list(f"Received t_step_no  : {t_step_no}")
        my_logger.add_to_wait_list(f"Received input_tensor shape: {input_tensor.shape}")
        
        
        normalized_input_tensor = my_dataset.normalize_strain_tensor(input_tensor)
            
        load_data_up_to_current = load_data_i[:t_step_no]            # True data used for training
        stored_strain_data_up_to_current = strain_data_i[:t_step_no] # True data used for training
        stored_output_data_up_to_current = output_data_i[:t_step_no] # True data used for training   
        my_logger.add_to_wait_list(f'load_data_up_to_current shape: {load_data_up_to_current.shape}')
        my_logger.add_to_wait_list(f'strain_data_up_to_current shape: {stored_strain_data_up_to_current.shape}')
        my_logger.add_to_wait_list(f'true_output_data_up_to_current shape: {stored_output_data_up_to_current.shape}')
        
        
        # Create/Update runtime_strain_data_up_to_current
        if t_step_no == 1:
            print('####### UPDATING runtime_strain_data_up_to_current')
            runtime_strain_data_up_to_current = torch.zeros_like(stored_strain_data_up_to_current)
            runtime_output_data_up_to_current = torch.zeros_like(stored_output_data_up_to_current)
        
        n_steps_accumulated =  runtime_strain_data_up_to_current.shape[0]
        if n_steps_accumulated == t_step_no:
            runtime_strain_data_up_to_current[-1] = normalized_input_tensor
        else:
            runtime_strain_data_up_to_current = torch.cat((runtime_strain_data_up_to_current, normalized_input_tensor), dim=0)
        
        my_logger.add_to_wait_list(f'runtime_strain_data_up_to_current shape: {runtime_strain_data_up_to_current.shape}')
        
        # Temp Check for the runtime_strain_data_up_to_current vs stored_strain_data_up_to_current
        strain_runtime_l2_norm_last = torch.norm(runtime_strain_data_up_to_current[-1])
        strain_stored_l2_norm_last = torch.norm(stored_strain_data_up_to_current[-1])
        strain_error_l2_norm_last = torch.norm(stored_strain_data_up_to_current[-1]-runtime_strain_data_up_to_current[-1])
        strain_rel_err_l2_norm_last = strain_error_l2_norm_last / strain_stored_l2_norm_last
        my_logger.add_to_wait_list(f'strain_error_l2_norm_last: {strain_error_l2_norm_last}')
        my_logger.add_to_wait_list(f'strain_rel_err_l2_norm_last: {strain_rel_err_l2_norm_last}')
        
        strain_runtime_l2_norm_full = torch.norm(runtime_strain_data_up_to_current)
        strain_stored_l2_norm_full = torch.norm(stored_strain_data_up_to_current)
        strain_error_l2_norm_full = torch.norm(stored_strain_data_up_to_current-runtime_strain_data_up_to_current)
        strain_rel_err_l2_norm_full = strain_error_l2_norm_full / strain_stored_l2_norm_full
        my_logger.add_to_wait_list(f'strain_error_l2_norm_full: {strain_error_l2_norm_full}')
        my_logger.add_to_wait_list(f'strain_rel_err_l2_norm_full: {strain_rel_err_l2_norm_full}')
        
        # Denormalized strain error check
        strain_runtime_l2_norm_last_denorm = torch.norm(my_dataset.denormalize_strain_tensor(runtime_strain_data_up_to_current[-1]))
        strain_stored_l2_norm_last_denorm = torch.norm(my_dataset.denormalize_strain_tensor(stored_strain_data_up_to_current[-1]))
        strain_error_l2_norm_last_denorm = torch.norm(my_dataset.denormalize_strain_tensor(stored_strain_data_up_to_current[-1])-my_dataset.denormalize_strain_tensor(runtime_strain_data_up_to_current[-1]))
        strain_rel_err_l2_norm_last_denorm = strain_error_l2_norm_last_denorm / strain_stored_l2_norm_last_denorm
        #####################################################################
        # Predicting Output through the model
        #####################################################################
        current_model_predicted_response = None
        

        if t_step_no > to_deeponet_link_step:
            to_use_strain_data_up_to_current = runtime_strain_data_up_to_current
        else:
            to_use_strain_data_up_to_current = stored_strain_data_up_to_current
        
        
        # create random noise tensor with the same shape as to_use_strain_data_up_to_current and make its range between -1 and 1
        if noise_method_index == 1:
            noise_tensor = torch.rand_like(to_use_strain_data_up_to_current) * 2 - 1
            noise_scaling_tensor = noise_tensor * noise_scale + 1
            print(f'max scaling factor: {torch.max(noise_scaling_tensor)}, min scaling factor: {torch.min(noise_scaling_tensor)}')
            to_use_strain_data_up_to_current = to_use_strain_data_up_to_current * noise_scaling_tensor  
        elif noise_method_index == 2:
            normalized_l2_norm = MDLH.get_l2_norm_by_sqrt_n(to_use_strain_data_up_to_current)
            noise_tensor = torch.rand_like(to_use_strain_data_up_to_current) * 2 - 1
            noise_updating_tensor = noise_tensor * noise_scale * normalized_l2_norm
            print(f'max noise_updating_tensor: {torch.max(noise_updating_tensor)}, min noise_updating_tensor: {torch.min(noise_updating_tensor)}')
            to_use_strain_data_up_to_current = to_use_strain_data_up_to_current + noise_updating_tensor
        
        with torch.no_grad():
            # add dimension for batch size in the input data
            current_model_predicted_response = model(load_data_up_to_current.unsqueeze(0).to(DEVICE),
                                                        to_use_strain_data_up_to_current.unsqueeze(0).to(DEVICE),
                                                        trunk_input_tensor.to(DEVICE)).squeeze(0).cpu()
        
        my_logger.add_to_wait_list(f'current_model_predicted_response shape: {current_model_predicted_response.shape}')
        
        if n_steps_accumulated == t_step_no:
            runtime_output_data_up_to_current[-1] = current_model_predicted_response[-1] 
        else:
            runtime_output_data_up_to_current = torch.cat((runtime_output_data_up_to_current, current_model_predicted_response[-1].unsqueeze(0)), dim=0)   
        
        my_logger.add_to_wait_list(f'runtime_output_data_up_to_current shape: {runtime_output_data_up_to_current.shape}')
        
        # compare current_model_predicted_response and runtime_output_data_up_to_current
        diff_current_model_vs_accumulated_runtime = torch.abs(current_model_predicted_response - runtime_output_data_up_to_current)
        # detect max difference location
        max_diff_current_model_vs_accumulated_runtime = torch.max(diff_current_model_vs_accumulated_runtime)
        max_diff_current_model_vs_accumulated_runtime_index = torch.argmax(diff_current_model_vs_accumulated_runtime).item()
        # get the max difference value
        max_diff_current_model_vs_accumulated_runtime_index_dim0 = max_diff_current_model_vs_accumulated_runtime_index // diff_current_model_vs_accumulated_runtime.shape[1]
        max_diff_current_model_vs_accumulated_runtime_index_dim1 = max_diff_current_model_vs_accumulated_runtime_index % diff_current_model_vs_accumulated_runtime.shape[1]
        print(f'diff_current_model_vs_accumulated_runtime max_diff_current_model_vs_accumulated_runtime: {max_diff_current_model_vs_accumulated_runtime}, index0: {max_diff_current_model_vs_accumulated_runtime_index_dim0}, index1: {max_diff_current_model_vs_accumulated_runtime_index_dim1}')
        
        
        
        output_predicted_l2_norm_last = torch.norm(runtime_output_data_up_to_current[-1])
        output_stored_l2_norm_last = torch.norm(stored_output_data_up_to_current[-1])
        output_error_l2_norm_last = torch.norm(stored_output_data_up_to_current[-1] - runtime_output_data_up_to_current[-1])
        output_rel_err_l2_norm_last = output_error_l2_norm_last / output_stored_l2_norm_last
        my_logger.add_to_wait_list(f'output_abs_diff_l2_norm_last: {output_error_l2_norm_last}')
        my_logger.add_to_wait_list(f'output_rel_diff_l2_norm_last: {output_rel_err_l2_norm_last}')
        
        output_predicted_l2_norm_full = torch.norm(runtime_output_data_up_to_current)
        output_stored_l2_norm_full = torch.norm(stored_output_data_up_to_current)
        output_error_l2_norm_full = torch.norm(stored_output_data_up_to_current - runtime_output_data_up_to_current)
        output_rel_err_l2_norm_full = output_error_l2_norm_full / output_stored_l2_norm_full
        my_logger.add_to_wait_list(f'output_abs_diff_l2_norm_full: {output_error_l2_norm_full}')
        my_logger.add_to_wait_list(f'output_rel_diff_l2_norm_full: {output_rel_err_l2_norm_full}')
        
        # Denormalized output error check
        output_predicted_l2_norm_last_denorm = torch.norm(my_dataset.denormalize_output_tensor(runtime_output_data_up_to_current[-1]))
        output_stored_l2_norm_last_denorm = torch.norm(my_dataset.denormalize_output_tensor(stored_output_data_up_to_current[-1]))
        output_error_l2_norm_last_denorm = torch.norm(my_dataset.denormalize_output_tensor(stored_output_data_up_to_current[-1]) - my_dataset.denormalize_output_tensor(runtime_output_data_up_to_current[-1]))
        output_rel_err_l2_norm_last_denorm = output_error_l2_norm_last_denorm / output_stored_l2_norm_last_denorm

        #####################################################################
        # End of predicting Output through the model
        #####################################################################
        
        
        # Output vector is the last element of the runtime_output_data_up_to_current
        output_tensor = runtime_output_data_up_to_current[-1]
        my_logger.add_to_wait_list(f"output_tensor shape: {output_tensor.shape}")
        
        my_logger.to_csv(f'target_load_case_index , {p_dict["target_load_case_index"]} , '
                         f't_step_no , {t_step_no} , '
                         f'strain_runtime_l2_norm_last , {strain_runtime_l2_norm_last} , '
                         f'strain_stored_l2_norm_last , {strain_stored_l2_norm_last} , '
                         f'strain_error_l2_norm_last , {strain_error_l2_norm_last} , '
                         f'strain_rel_err_l2_norm_last , {strain_rel_err_l2_norm_last} , '
                         f'output_predicted_l2_norm_last , {output_predicted_l2_norm_last} , '
                         f'output_stored_l2_norm_last , {output_stored_l2_norm_last} , '
                         f'output_error_l2_norm_last , {output_error_l2_norm_last} , '
                         f'output_rel_err_l2_norm_last , {output_rel_err_l2_norm_last} , ')
        
        p_dict['stored_errors_array'].append([load_case_index, t_step_no, strain_runtime_l2_norm_last, strain_stored_l2_norm_last, strain_error_l2_norm_last, strain_rel_err_l2_norm_last, output_predicted_l2_norm_last, output_stored_l2_norm_last, output_error_l2_norm_last, output_rel_err_l2_norm_last])    
        p_dict['stored_errors_array_full'].append([load_case_index, t_step_no, strain_runtime_l2_norm_full, strain_stored_l2_norm_full, strain_error_l2_norm_full, strain_rel_err_l2_norm_full, output_predicted_l2_norm_full, output_stored_l2_norm_full, output_error_l2_norm_full, output_rel_err_l2_norm_full])
        p_dict['stored_errors_array_denorm'].append([load_case_index, t_step_no, strain_runtime_l2_norm_last_denorm, strain_stored_l2_norm_last_denorm, strain_error_l2_norm_last_denorm, strain_rel_err_l2_norm_last_denorm, output_predicted_l2_norm_last_denorm, output_stored_l2_norm_last_denorm, output_error_l2_norm_last_denorm, output_rel_err_l2_norm_last_denorm])

        output_tensor_to_send = None
        if t_step_no > to_dealii_link_step:
            output_tensor_to_send = runtime_output_data_up_to_current[-1]
        else:
            output_tensor_to_send = stored_output_data_up_to_current[-1]
            
        # Denormalize the output_tensor_to_send
        denormalized_output_tensor = my_dataset.denormalize_output_tensor(output_tensor_to_send)
        predicted_output_data_up_to_current_denormalized = my_dataset.denormalize_output_tensor(runtime_output_data_up_to_current)
        
        denormalized_output_tensor = denormalized_output_tensor.reshape(-1)
        # convert to tuple of floats
        output_vector = tuple(denormalized_output_tensor.tolist())
        
        # Write the output back to shared memory
        shared_output.seek(0)
        shared_output.write(struct.pack(shared_output_format, *output_vector))

        # Signal C++ that output is ready
        sem_python_output_ready.release()
        sem_solution_saved.acquire()
        my_logger.add_to_wait_list("One python process done ")
        my_logger.add_to_wait_list("Solution saved by C++ -----------------------------------------")
        
        my_logger.log_and_print_the_wait_list()
        
        if t_step_no == 100:
            simulation_end_time = time.time()
            simulation_time = simulation_end_time - simulation_start_time
            my_logger.log_and_print(f"Simulation Time in Minutes: {simulation_time/60}")
            # save p_dict['stored_errors_array'] as a numpy array
            stored_errors_array_np = np.array(p_dict['stored_errors_array'])
            stored_errors_array_full_np = np.array(p_dict['stored_errors_array_full'])
            stored_errors_array_denorm_np = np.array(p_dict['stored_errors_array_denorm'])
            # save arrays for future comparison
            init_save_name = log_file_name.replace('_LCX_', f'_LC{p_dict["target_load_case_index"]}_').replace('.log', '') 
            to_use_plots_save_folder = f'{plots_save_folder}/{init_save_name}'
            to_use_npy_save_folder = f'{npy_save_folder}/{init_save_name}'
            # ensure folder exists
            ensure_folder_exists(to_use_plots_save_folder)
            ensure_folder_exists(to_use_npy_save_folder)
            
            
            errors_save_name = init_save_name +'_errors.npy'
            np.save(f'{to_use_npy_save_folder}/{errors_save_name}', stored_errors_array_np)
            errors_save_name_full = init_save_name +'_errors_full.npy'
            np.save(f'{to_use_npy_save_folder}/{errors_save_name_full}', stored_errors_array_full_np)
            errors_save_name_denorm = init_save_name +'_errors_denorm.npy'
            np.save(f'{to_use_npy_save_folder}/{errors_save_name_denorm}', stored_errors_array_denorm_np)
            my_logger.log_and_print("stored_errors_array_np saved as a numpy array")
            # plot the errors
            MSLH.plot_data_for_error_array(stored_errors_array_np, used_model_name, to_dealii_link_step, to_deeponet_link_step, noise_scale, noise_method_index, to_use_plots_save_folder, 'a-errors')
            MSLH.plot_data_for_error_array(stored_errors_array_full_np, used_model_name, to_dealii_link_step, to_deeponet_link_step, noise_scale, noise_method_index, to_use_plots_save_folder, 'a-errors_full')
            MSLH.plot_data_for_error_array(stored_errors_array_denorm_np, used_model_name, to_dealii_link_step, to_deeponet_link_step, noise_scale, noise_method_index, to_use_plots_save_folder, 'a-errors_denorm')
            # load deal.ii saved data
            args_load_case_index = p_dict["target_load_case_index"]
            args_load_case_file_name_lc = load_case_file_name.replace('lcxxx',f'load{args_load_case_index}')
            all_steps_name = args_load_case_file_name_lc.replace('stxxx','all-steps')
            time_step_list = np.arange(1, 101, 1)
            n_time_steps = len(time_step_list)
            dealii_disp_array_grid = np.zeros((n_time_steps, 31, 31, 31, 3))
            for ii, time_index in enumerate(time_step_list):
                my_logger.log_and_print(f'processing load_case_no: {args_load_case_index}, time_index: {time_index}/{n_time_steps}')
                data_array_i, _, _, _ = MSLH.custom_get_data(args_load_case_file_name_lc, time_index, False, False, fem_readpath)
                dealii_disp_array_grid[ii, :, :, :, :] = data_array_i
            print(f'shape of predicted_output_data_up_to_current_denormalized: {predicted_output_data_up_to_current_denormalized.shape}')
            deepOnet_theta_array_grid = predicted_output_data_up_to_current_denormalized.cpu().numpy().reshape(100,31,31,31,1)
            # concatenate the two arrays
            combined_predicted_array = np.concatenate((deepOnet_theta_array_grid, dealii_disp_array_grid), axis=-1)
            # load true Data
            full_data_dict = p_dict['full_data_dict']
            output_data_i_full_denormalized = full_data_dict['output_data_i_full_denormalized']
            combined_true_array = output_data_i_full_denormalized.reshape(100, 31, 31, 31, 4).cpu().numpy()
            save_name_true = init_save_name +'_true.npy' 
            save_name_pred = init_save_name +'_pred.npy'
            np.save(f'{to_use_npy_save_folder}/{save_name_true}', combined_true_array)
            np.save(f'{to_use_npy_save_folder}/{save_name_pred}', combined_predicted_array)
            # plot true vs predicted slices
            
            MDLH.plot_true_vs_prediction_3D(combined_true_array, combined_predicted_array, p_dict["target_load_case_index"], init_save_name, 'I-FENN', to_use_plots_save_folder)
            MDLH.plot_true_vs_prediction_slice(combined_true_array, combined_predicted_array, p_dict["target_load_case_index"], init_save_name, 'I-FENN', to_use_plots_save_folder)
            # ensure       
            cleanup()
            exit()

def cleanup():
    # Unpack the shared memories
    shared_flags = shm_object_dict["shared_flags"]
    shared_input = shm_object_dict["shared_input"]
    shared_output = shm_object_dict["shared_output"]
    shared_check = shm_object_dict["shared_check"]
    shm_fd0 = shm_object_dict["shm_fd0"]
    shm_fd1 = shm_object_dict["shm_fd1"]
    shm_fd2 = shm_object_dict["shm_fd2"]
    shm_fd3 = shm_object_dict["shm_fd3"]
    sem_python_output_ready = shm_object_dict["sem_python_output_ready"]
    sem_cpp_input_ready = shm_object_dict["sem_cpp_input_ready"]
    sem_solution_saved = shm_object_dict["sem_solution_saved"]

    # Delete shared memories here
    shared_flags.close()
    shared_input.close()
    shared_output.close()
    shared_check.close()
    shm_fd0.unlink()
    shm_fd1.unlink()
    shm_fd2.unlink()
    shm_fd3.unlink()
    sem_python_output_ready.unlink()
    sem_cpp_input_ready.unlink()
    sem_solution_saved.unlink()
    print("Cleanup done")    

def signal_handler(sig, frame):
    print('Interrupt received, cleaning up...')
    cleanup()
    sys.exit(0)
    
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
class InputOptions(Enum):
    Load = 1
    Strain = 2
    Load_Strain = 3
   
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")   
    
def load_data_for_a_specific_load_case(load_case_index):
    load_folder_path   = f'../../2-Processed_Generated_Data'
    output_folder_path = f'../../2-Processed_Generated_Data'
    strain_folder_path = f'../../2-Processed_Generated_Data'
    
    load_file_name = f'Grid_3D_BodySource_V004_force_per_history_increment_grid_LC-'
    output_file_name = f'all_data_array_grid_LC-'
    strain_file_name = f'strain-center_data_array_grid_LC-'

    coordinates_file_name = f'coordinates_array_grid_LC-0-100'
    normalization_option = 2 # 0: no normalization, 1: normalize between 0 and 1, 2: normalize between -1 and 1 3: standardization 4: divide by max value 5: divide by std
    loss_criterion_option = 7 # 1 : MSELoss('mean') 2 : L1Loss('mean') 3 : SmoothL1Loss('mean') 4 : MSELoss('sum') 5 : L1Loss('sum') 6 : SmoothL1Loss('sum') 7 : CustomLoss (L2) 8 : CustomLoss (Normalized L2)

    LC_start = load_case_index
    LC_end = load_case_index + 1
    max_items_saved = 1
    #--------------------------------------------------------------------------------
    dl_batch_size = 16
    selected_input_option = InputOptions.Load_Strain
    split_in_branch = True
    enforce_BC = True
    train_for_theta_only = True
    read_all_the_file_once = False # (True Preferred for HPC) works only if max_items_saved is > (LC_end - LC_start)
    use_torch_multiprocessing_manager = False
    epochs_sessions_list = [10,30,60,4000-100,4000,4000,4000,4000,4000]
    device_save_name = 'dev1'
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
                    'point_output_skip': (1,1,1),
                    'point_output_end': (None,None,None), #(None,None,None),
                    'point_output_flatten': True}
    main_cashed_items_dict = {}
    test_cashed_items_dict = {}

    # Create logger object
    for filepath in [logs_save_folder, plots_save_folder, models_save_folder]:
        ensure_folder_exists(filepath)
    to_use_file_name = log_file_name.replace('_LCX_', f'_LC{load_case_index}_') 
    my_logger = MDLH.MyLogger(f'./{logs_save_folder}/{to_use_file_name}', max_wait_list_length = 50, use_csv=True)
    p_dict['my_logger'] = my_logger
    #_%%
    # Set as global variable in the helper module
    MDLH.main_cashed_items_dict = main_cashed_items_dict
    MDLH.test_cashed_items_dict = test_cashed_items_dict
    MDLH.use_torch_multiprocessing_manager = use_torch_multiprocessing_manager
    MDLH.my_logger = my_logger
    MDLH.plots_save_folder = plots_save_folder
    MDLH.models_save_folder = models_save_folder
    MDLH.train_for_theta_only = train_for_theta_only

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

    my_dataset = MDLH.MyDataset(data_loading_settings, mode = 'test')
    print(f'length of my_dataset: {len(my_dataset)}')
    load_data_i, strain_data_i, output_data_i = my_dataset[0]
    print(f'load_data_[0] shape: {load_data_i.shape}')
    print(f'strain_data_[0] shape: {strain_data_i.shape}')
    print(f'output_data_[0] shape: {output_data_i.shape}')
    print(f'load_data_[0]=> max: {torch.max(load_data_i)}, min: {torch.min(load_data_i)}, mean: {torch.mean(load_data_i)}, std: {torch.std(load_data_i)}')
    print(f'strain_data_[0]=> max: {torch.max(strain_data_i)}, min: {torch.min(strain_data_i)}, mean: {torch.mean(strain_data_i)}, std: {torch.std(strain_data_i)}')
    print(f'output_data_[0][0]=> max: {torch.max(output_data_i[...,0])} , min: {torch.min(output_data_i[...,0])}, mean: {torch.mean(output_data_i[...,0])}, std: {torch.std(output_data_i[...,0])}')
    
    load_data_i_shape = load_data_i.shape
    strain_data_i_shape = strain_data_i.shape
    output_data_i_shape = output_data_i.shape
    
    ## Load Full Data
    MDLH.train_for_theta_only = False # temporary change to load data
    
    my_dataset_full = MDLH.MyDataset(data_loading_settings, mode = 'train')
    print(f'length of my_dataset_full: {len(my_dataset_full)}')
    load_data_i_full, strain_data_i_full, output_data_i_full = my_dataset_full[0]
    print(f'load_data_i_full shape: {load_data_i_full.shape}')
    print(f'strain_data_i_full shape: {strain_data_i_full.shape}')
    print(f'output_data_i_full shape: {output_data_i_full.shape}')
    print(f'load_data_i_full=> max: {torch.max(load_data_i_full)}, min: {torch.min(load_data_i_full)}, mean: {torch.mean(load_data_i_full)}, std: {torch.std(load_data_i_full)}')
    print(f'strain_data_i_full=> max: {torch.max(strain_data_i_full)}, min: {torch.min(strain_data_i_full)}, mean: {torch.mean(strain_data_i_full)}, std: {torch.std(strain_data_i_full)}')
    print(f'output_data_i_full[0]=> max: {torch.max(output_data_i_full[...,0])} , min: {torch.min(output_data_i_full[...,0])}, mean: {torch.mean(output_data_i_full[...,0])}, std: {torch.std(output_data_i_full[...,0])}')
    print(f'output_data_i_full[1]=> max: {torch.max(output_data_i_full[...,1])} , min: {torch.min(output_data_i_full[...,1])}, mean: {torch.mean(output_data_i_full[...,1])}, std: {torch.std(output_data_i_full[...,1])}')
    print(f'output_data_i_full[2]=> max: {torch.max(output_data_i_full[...,2])} , min: {torch.min(output_data_i_full[...,2])}, mean: {torch.mean(output_data_i_full[...,2])}, std: {torch.std(output_data_i_full[...,2])}')
    print(f'output_data_i_full[3]=> max: {torch.max(output_data_i_full[...,3])} , min: {torch.min(output_data_i_full[...,3])}, mean: {torch.mean(output_data_i_full[...,3])}, std: {torch.std(output_data_i_full[...,3])}')
    
    output_data_i_full_denormalized = my_dataset_full.denormalize_output_tensor(output_data_i_full)
    
    full_data_dict = {  
        'load_data_i_full': load_data_i_full,
        'strain_data_i_full': strain_data_i_full,
        'output_data_i_full': output_data_i_full,
        'output_data_i_full_denormalized': output_data_i_full_denormalized,
    }
    
    ###
    MDLH.train_for_theta_only = train_for_theta_only
    
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
            # self.dropout2 = nn.Dropout(p=general_dropout)  # Dropout layer
            # self.fc2 = nn.Linear(output_dim, output_dim)

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

    
    return load_data_i, strain_data_i, output_data_i, my_dataset, model, DEVICE, full_data_dict

def load_checkpoint_into_pytorch_model(model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    
    print(f" ## Model loaded from checkpoint: {checkpoint_path}")
    
    return model

def update_pytorch_dict_for_load_case(load_case_index):
    load_data_i, strain_data_i, output_data_i, my_dataset, model, DEVICE, full_data_dict = load_data_for_a_specific_load_case(load_case_index)
    model = load_checkpoint_into_pytorch_model(model)
    trunk_input_tensor = my_dataset.get_coordinates_array(return_as_tensor=True)
    
    p_dict['target_load_case_index'] = load_case_index
    p_dict['load_data_i'] = load_data_i
    p_dict['strain_data_i'] = strain_data_i
    p_dict['output_data_i'] = output_data_i
    p_dict['my_dataset'] = my_dataset
    p_dict['model'] = model
    p_dict['DEVICE'] = DEVICE
    p_dict['trunk_input_tensor'] = trunk_input_tensor
    p_dict['full_data_dict'] = full_data_dict
    
    p_dict['stored_errors_array'] = []
    p_dict['stored_errors_array_full'] = []
    p_dict['stored_errors_array_denorm'] = []
    
############################################################################################################
# Main code implementation
############################################################################################################
# declare global variable to store shared memories
shm_object_dict = {} 
load_case_data = {}
p_dict = {}
noise_scale = 0.0
noise_method_index = int(1)
to_dealii_link_step = 0
to_deeponet_link_step = 0
fem_readpath = "../FEM_CPP_Layer/solution_ifenn"
load_case_file_name = 'Model009_V015-tests_lcxxx_sc2.0e-03_mesh30_t18000_inc100_dt180_stxxx.h5'

logs_save_folder = 'logs'
plots_save_folder = 'plots'
npy_save_folder = 'npy_files'
models_save_folder = 'models'
used_model_name = 'T001'
log_file_name = f'run_000_{used_model_name}_LCX_{to_dealii_link_step}_{to_deeponet_link_step}_{noise_scale}_{noise_method_index}.log'
checkpoint_path = f'./models/trained_model.pth'

simulation_start_time = None
simulation_end_time = None
runtime_strain_data_up_to_current = None
runtime_output_data_up_to_current = None

# load shared memories
load_shared_memories()

start_input_out_process()