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

def collect_structured_data_from_h5(file_path, keys, variable_names, variable_indices, max_coordinates, n_cell_per_directions, print_info):
    # Open the HDF5 file
    h5_data_dict = read_data_from_h5(file_path, keys)  


    coordinates = h5_data_dict['nodes']
    
    if print_info:
        unique_x_values = np.sort(np.unique(coordinates[:,0]))
        unique_y_values = np.sort(np.unique(coordinates[:,1]))
        unique_z_values = np.sort(np.unique(coordinates[:,2]))
        print(f'nodes increments x: {len(unique_x_values)}, y: {len(unique_y_values)}, z: {len(unique_z_values)}')

    n_cell_per_direction_x = n_cell_per_directions[0]
    n_cell_per_direction_y = n_cell_per_directions[1]
    n_cell_per_direction_z = n_cell_per_directions[2]
    max_x = max_coordinates[0]
    max_y = max_coordinates[1]
    max_z = max_coordinates[2]
    x_increment = max_x/n_cell_per_direction_x
    y_increment = max_y/n_cell_per_direction_y
    z_increment = max_z/n_cell_per_direction_z

    n_nodes_per_direction_x = n_cell_per_direction_x + 1
    n_nodes_per_direction_y = n_cell_per_direction_y + 1
    n_nodes_per_direction_z = n_cell_per_direction_z + 1
    
    n_variables = len(variable_names)
    n_nodes_actual = len(coordinates)

    data_array_grid = np.zeros((n_nodes_per_direction_x, n_nodes_per_direction_y, n_nodes_per_direction_z, n_variables))
    coordinates_array_grid = np.zeros((n_nodes_per_direction_x, n_nodes_per_direction_y, n_nodes_per_direction_z, 3))
    # create boolean array to check if node is found
    is_node_found = np.zeros((n_nodes_per_direction_x, n_nodes_per_direction_y, n_nodes_per_direction_z), dtype=bool)
    # create array to store the count of nodes found
    node_count_per_index = np.zeros((n_nodes_per_direction_x, n_nodes_per_direction_y, n_nodes_per_direction_z), dtype=int)


    # loop over all nodes and collect data
    for i_node in range(n_nodes_actual):
        node_coord = coordinates[i_node]
        x = node_coord[0]
        y = node_coord[1]
        z = node_coord[2]
        # find the index of the node
        x_index = np.round(x/x_increment).astype(int)
        y_index = np.round(y/y_increment).astype(int)
        z_index = np.round(z/z_increment).astype(int)
        # store the coordinates
        coordinates_array_grid[x_index, y_index, z_index, :] = node_coord
        # store the data
        for j in range(n_variables):
            data_array_grid[x_index, y_index, z_index, j] = h5_data_dict[variable_names[j]][i_node][variable_indices[j]]
        # update the boolean array
        is_node_found[x_index, y_index, z_index] = True
        node_count_per_index[x_index, y_index, z_index] += 1

    # check if all nodes are found (all values in the boolean array are True)
    if np.all(is_node_found):
        if print_info:
            print('All nodes are found')
    else:   
        if print_info:
            print('Not all nodes are found')
            print(f'Number of nodes not found: {np.sum(~is_node_found)}')
        raise ValueError('Not all nodes are found')
    
    if print_info:
        # Check repitition of nodes    
        max_repeated_nodes = np.max(node_count_per_index)
        print(f'Maximum number of nodes found at a single index: {max_repeated_nodes}')    
    
    return data_array_grid,coordinates_array_grid, is_node_found, node_count_per_index


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
def custom_get_data(load_case_file_name, time_index, plot_repition_map_required , print_info, pre_file_path):
    n_cell_per_direction = 30
    target_time_index = time_index
    target_time_index_zero_padded = f'{target_time_index:04}'
    current_case_file_name = load_case_file_name.replace('stxxx',f'{target_time_index_zero_padded}')

    n_cell_per_directions = [n_cell_per_direction,n_cell_per_direction,n_cell_per_direction]
    file_path = f'{pre_file_path}/{current_case_file_name}'
    target_keys = ['nodes', 'displacement']  
    max_coordinates = [1.0, 1.0, 1.0]
    variable_names = ['displacement', 'displacement', 'displacement']
    variable_indices = [0, 1, 2]
    data_array_grid,coordinates_array_grid, is_node_found, node_count_per_index = collect_structured_data_from_h5(file_path, target_keys, variable_names, variable_indices, max_coordinates, n_cell_per_directions, print_info)
    
    if plot_repition_map_required:
        plot_repititions_histogram(node_count_per_index)
        plot_repition_map(node_count_per_index,0,int(n_cell_per_directions[0]/2))
        plot_repition_map(node_count_per_index,1,int(n_cell_per_directions[1]/2))
        plot_repition_map(node_count_per_index,2,int(n_cell_per_directions[2]/2))
    
    return data_array_grid,coordinates_array_grid, is_node_found, node_count_per_index
    
def plot_data_for_error_array(error_data, model_name, to_dealii_link_step, to_deeponet_link_step, noise_scale, noise_method_index, plots_save_folder, data_type_name):
    n_time_steps = error_data.shape[0]
    time_step_values = error_data[:,1]
    strain_err_values = error_data[:,5]
    theta_err_values = error_data[:,9]
    # print(time_step_values)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_step_values, strain_err_values, label='Strain trace')
    ax.plot(time_step_values, theta_err_values, label='Temperature')
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