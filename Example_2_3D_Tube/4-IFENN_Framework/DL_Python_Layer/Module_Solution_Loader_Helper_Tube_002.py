import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from matplotlib.ticker import ScalarFormatter

def explore_h5_file(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        # Explore the file structure
        def explore_h5_group(group, indent=0):
            for key in group.keys():
                item = group[key]
                print('  ' * indent + key)
                if isinstance(item, h5py.Group):
                    explore_h5_group(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    print('  ' * (indent + 1) + f'Dataset: {item.shape}, {item.dtype}')

        explore_h5_group(h5_file)

def read_data_from_h5(file_path, keys):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        # Explore the file structure
        if keys is None:
            keys = h5_file.keys()
            
        data_dict = {}    
        for key in keys:
            item = h5_file[key]
            data_dict[key] = np.array(item)
        return data_dict    

def arctan2_positive(y, x):
    angle = np.arctan2(y, x)  # Get the angle in the range [-π, π]
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)  # Convert to [0, 2π]
    tolerance = 2*np.pi/1e6
    angle[np.isclose(angle, 2*np.pi, atol=tolerance)] = 0.0
    return np.where(angle < 0, angle + 2 * np.pi, angle)  # Convert to [0, 2π]

def get_coords_arz(coords_xyz):
    coords_x = coords_xyz[:,0]
    coords_y = coords_xyz[:,1]
    coords_z = coords_xyz[:,2]
    coords_angle = arctan2_positive(coords_y, coords_x) # Get the angle in the range [0, 2π]
    coords_r = np.sqrt(coords_x**2 + coords_y**2)
    return coords_angle, coords_r, coords_z

def collect_structured_data_from_h5(file_path, print_info):
    n_cell_per_angle = 8*8
    n_cell_per_r = 8
    n_cell_per_z = 8
    
    n_node_per_angle = n_cell_per_angle
    n_node_per_r = n_cell_per_r + 1
    n_node_per_z = n_cell_per_z + 1
    
    expected_n_nodes = n_node_per_angle * n_node_per_r * n_node_per_z
    expected_n_cells = n_cell_per_angle * n_cell_per_r * n_cell_per_z
    
    if print_info:
        print(f'---------------------------------------------------------')
        print(f'Expected number of nodes: {expected_n_nodes}')
        print(f'Expected number of cells: {expected_n_cells}')
    
    # Open the HDF5 file
    file_keys = ['nodes', 'displacement', 'cells']  
    h5_data_dict = read_data_from_h5(file_path, file_keys)  
    # print(f'File Loaded: {file_path}')

    coords_xyz = h5_data_dict['nodes']
    
    coords_x = coords_xyz[:,0]
    coords_y = coords_xyz[:,1]
    coords_z = coords_xyz[:,2]
    
    coords_angle, coords_r, coords_z = get_coords_arz(coords_xyz)
    
    min_angle = 0.0
    max_angle = 2*np.pi
    min_r = 1.0    
    max_r = 2.0
    min_z = 0.0
    max_z = 1.0
    increment_angle = (max_angle - min_angle)/n_cell_per_angle
    increment_r = (max_r - min_r)/n_cell_per_r
    increment_z = (max_z - min_z)/n_cell_per_z
    
    coords_z_index = np.round((coords_z - min_z)/increment_z).astype(int)
    coords_r_index = np.round((coords_r - min_r)/increment_r).astype(int)
    coords_angle_index = np.round((coords_angle - min_angle)/increment_angle).astype(int)
    # if angle_index == n_node_per_angle, then angle_index = 0
    coords_angle_index[coords_angle_index == n_node_per_angle] = 0
    
    if print_info:
        print(f'---------------------------------------------------------')
        print(f'max x: {np.max(coords_x)} min x: {np.min(coords_x)}')
        print(f'max y: {np.max(coords_y)} min y: {np.min(coords_y)}')
        print(f'max z: {np.max(coords_z)} min z: {np.min(coords_z)}')
        print(f'max angle: {np.max(coords_angle)} min angle: {np.min(coords_angle)}')
        print(f'max r: {np.max(coords_r)} min r: {np.min(coords_r)}')
    
    if print_info:
        print(f'---------------------------------------------------------')
        unique_coords_z_index = np.unique(coords_z_index)
        unique_coords_r_index = np.unique(coords_r_index)
        unique_coords_angle_index = np.unique(coords_angle_index)

        print(f'Number of unique angle indices: {len(unique_coords_angle_index)}')
        print(f'Number of unique r indices: {len(unique_coords_r_index)}')  
        print(f'Number of unique z indices: {len(unique_coords_z_index)}')
    
    
    if print_info:
        print(f'---------------------------------------------------------')
        ## test to get back the coords_xyz from the indices
        test_coords_z = min_z + coords_z_index*increment_z
        test_coords_r = min_r + coords_r_index*increment_r
        test_coords_angle = min_angle + coords_angle_index*increment_angle
        test_coords_x = test_coords_r*np.cos(test_coords_angle)
        test_coords_y = test_coords_r*np.sin(test_coords_angle)
        test_coords = np.column_stack((test_coords_x, test_coords_y, test_coords_z))
        erorr_coords = coords_xyz - test_coords
        max_error = np.max(np.abs(erorr_coords))
        print(f'Maximum error in coordinates: {max_error}')       
    
    if print_info:
        print(f'---------------------------------------------------------')
        unique_z_values = np.sort(np.unique(coords_xyz[:,2]))
        round_coords_angle = np.round(coords_angle, 5)
        round_coords_r = np.round(coords_r, 5)
        unique_angle_values = np.sort(np.unique(round_coords_angle))
        unique_r_values = np.sort(np.unique(round_coords_r))
        print(f'Number of unique angle values: {len(unique_angle_values)}') 
        print(f'Number of unique r values: {len(unique_r_values)}')
        print(f'Number of unique z values: {len(unique_z_values)}')
        
    variable_names = ['displacement', 'displacement', 'displacement']
    variable_indices = [ 0, 1, 2]

    data_grid = np.zeros((n_node_per_angle, n_node_per_r, n_node_per_z, len(variable_names)))
    data_count_grid = np.zeros((n_node_per_angle, n_node_per_r, n_node_per_z), dtype=int)
    coords_grid = np.zeros((n_node_per_angle, n_node_per_r, n_node_per_z, 3))
    
    for i_node in range(len(coords_xyz)):
        angle_index = coords_angle_index[i_node]
        r_index = coords_r_index[i_node]
        z_index = coords_z_index[i_node]
        for j in range(len(variable_names)):
            data_grid[angle_index, r_index, z_index, j] = h5_data_dict[variable_names[j]][i_node][variable_indices[j]]
        data_count_grid[angle_index, r_index, z_index] += 1
        coords_grid[angle_index, r_index, z_index, :] = coords_xyz[i_node]
    
    is_node_found_grid = data_count_grid > 0
    
    # check if all nodes are found (all values in the boolean array are True)
    if np.all(is_node_found_grid):
        if print_info:
            print('## All nodes are found')
    else:   
        if print_info:
            print('Not all nodes are found')
            print(f'Number of nodes not found: {np.sum(~is_node_found_grid)}')
        raise ValueError('Not all nodes are found')
    
    if print_info:
        # Check repitition of nodes    
        max_repeated_nodes = np.max(data_count_grid)
        print(f'Maximum number of nodes found at a single index: {max_repeated_nodes}')

    return data_grid, coords_grid, data_count_grid    


def plot_repititions_histogram(node_count_per_index):
    # plot histogram of repititions
    plt.figure()
    plt.hist(node_count_per_index.flatten(), bins=np.arange(1, np.max(node_count_per_index)+2)-0.5, edgecolor='black')
    plt.xlabel('Number of nodes found at a single index')
    plt.ylabel('Frequency')
    plt.title('Histogram of repititions of nodes')
    plt.grid()
    plt.show()
    
def plot_repition_map(node_count_per_index,axis_no,axis_index):
    # plot the repitition map
    plt.figure()
    if axis_no == 0:
        plt.imshow(node_count_per_index[axis_index, :, :].T, origin='lower', cmap='viridis')
        plt.xlabel('y index')
        plt.ylabel('z index')
        plt.title(f'Repitition map at x index {axis_index}')
    elif axis_no == 1:
        plt.imshow(node_count_per_index[:, axis_index, :].T, origin='lower', cmap='viridis')
        plt.xlabel('x index')
        plt.ylabel('z index')
        plt.title(f'Repitition map at y index {axis_index}')
    elif axis_no == 2:
        plt.imshow(node_count_per_index[:, :, axis_index].T, origin='lower', cmap='viridis')
        plt.xlabel('x index')
        plt.ylabel('y index')
        plt.title(f'Repitition map at z index {axis_index}')
    plt.colorbar()
    plt.show()    

##%%
def custom_get_data(file_name, time_index, print_info, pre_file_path):
    # target_load_index = load_case
    target_time_index = time_index
    target_time_index_zero_padded = f'{target_time_index:04}'
    file_name = file_name.replace('stxxx', target_time_index_zero_padded)
    file_path = f'{pre_file_path}/{file_name}'
    
    data_grid, coords_grid, _ = collect_structured_data_from_h5(file_path, print_info)
    
    return data_grid, coords_grid

   
def plot_data_for_error_array(error_data, model_name, to_dealii_link_step, to_deeponet_link_step, noise_scale, noise_method_index, plots_save_folder, data_type_name):
    n_time_steps = error_data.shape[0]
    time_step_values = error_data[:,1]
    strain_err_values = error_data[:,5]
    theta_err_values = error_data[:,9]
    # print(time_step_values)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_step_values, strain_err_values, label='deal.ii -> Strain')
    ax.plot(time_step_values, theta_err_values, label='DeepONet -> Theta')
    # set the x-axis label
    ax.set_xlabel('Time Step')
    # set the y-axis label
    ax.set_ylabel('Error Normalized L2 Norm')
    # Use log scale for the y-axis
    ax.set_yscale('log')
    # Show the legend
    ax.legend()
    title_0 = f'model: {model_name} '
    title_1 = f'link to dealii after {to_dealii_link_step} steps'
    title_2 = f'link to DeepONet after {to_deeponet_link_step} steps'  
    title_3 = f'noise scale {noise_scale}, method {int(noise_method_index)}'  
    title = title_0 + ' \n ' + title_1 + ' \n ' + title_2  + ' \n ' + title_3  
    # set a title
    ax.set_title(title)
    # set y axis limits
    ax.set_ylim([1e-7, 1e+1])
    # set y axis ticks
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    # Set the y-axis to have ticks every 10^1 increment
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), numticks=10))
    # show major horizontal gridlines
    ax.yaxis.grid(True, which='major')
    
    plt.tight_layout()  # This automatically adjusts the spacing
    
    # save the plot
    plt.savefig(f'./{plots_save_folder}/{data_type_name}.png')
        
    try:
        plt.close(fig) 
    except:
        pass