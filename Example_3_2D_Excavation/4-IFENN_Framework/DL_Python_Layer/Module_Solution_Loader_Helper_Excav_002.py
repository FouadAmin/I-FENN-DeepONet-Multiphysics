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

def collect_structured_data_from_h5(file_path, compute_maps, print_info):
    y_direction_length = 40
    x_direction_length = 60
    cell_size = 1.0
    cells_per_x_dir = int(np.round(x_direction_length/cell_size))
    cells_per_y_dir = int(np.round(y_direction_length/cell_size))

    n_unique_cells = 2200
    n_unique_nodes = 2309

    nodes_per_x_dir = cells_per_x_dir + 1
    nodes_per_y_dir = cells_per_y_dir + 1

    coords_per_node = 2
    coords_per_cell = 2
    response_values_per_node = 2
    response_values_per_cell = 1


    # To be filled during data loading
    grid_nodes_exist = np.full((nodes_per_x_dir, nodes_per_y_dir), False)
    grid_nodes_count = np.zeros((nodes_per_x_dir, nodes_per_y_dir), dtype=int)
    grid_nodes_response = np.full((nodes_per_x_dir, nodes_per_y_dir, response_values_per_node), np.nan, dtype=np.float64) 
    grid_nodes_coords = np.full((nodes_per_x_dir, nodes_per_y_dir, coords_per_node), np.nan, dtype=np.float64)

    grid_cells_exist = np.full((cells_per_x_dir, cells_per_y_dir), False)
    grid_cells_count = np.zeros((cells_per_x_dir, cells_per_y_dir), dtype=int)
    grid_cells_response = np.full((cells_per_x_dir, cells_per_y_dir, response_values_per_cell), np.nan, dtype=np.float64)
    grid_cells_coords = np.full((cells_per_x_dir, cells_per_y_dir, coords_per_cell), np.nan, dtype=np.float64)

    '''
    File Structure:
    cell_central_strain_trace
      Dataset: (8800, 1), float64
    cells
      Dataset: (2200, 4), uint32
    displacement_x
      Dataset: (8800, 1), float64
    displacement_y
      Dataset: (8800, 1), float64
    nodes
      Dataset: (8800, 2), float64
    pressure
      Dataset: (8800, 1), float64 
    '''
    
    # Open the HDF5 file
    # file_keys = ['cells', 'nodes', 'pressure', 'displacement_x', 'displacement_y', 'cell_central_strain_trace']  
    file_keys = ['cells', 'nodes', 'displacement_x', 'displacement_y']  
    h5_data_dict = read_data_from_h5(file_path, file_keys)  
    h5_cell_nodes = h5_data_dict['cells']
    h5_node_coords = h5_data_dict['nodes']
    h5_node_displacement_x = h5_data_dict['displacement_x']
    h5_node_displacement_y = h5_data_dict['displacement_y']
  
    n_cells = len(h5_cell_nodes)
    
    
    # Loop through the cells and fill the grid nodes
    for i_cell in range(n_cells):
        cell_node_indices = h5_cell_nodes[i_cell]
        cell_x_values = h5_node_coords[cell_node_indices, 0]
        cell_y_values = h5_node_coords[cell_node_indices, 1]
        cell_min_x = np.min(cell_x_values)
        cell_min_y = np.min(cell_y_values)
        cell_min_x_index = int(np.round(cell_min_x/cell_size))
        cell_min_y_index = int(np.round(cell_min_y/cell_size))

        if cell_min_x_index < 0 or cell_min_x_index >= cells_per_x_dir or cell_min_y_index < 0 or cell_min_y_index >= cells_per_y_dir:
            # raise error
            print(f'Cell index out of bounds: {cell_min_x_index}, {cell_min_y_index}')
            Exception('Cell index out of bounds')


        grid_cells_exist[cell_min_x_index, cell_min_y_index] = True
        grid_cells_count[cell_min_x_index, cell_min_y_index] += 1
        grid_cells_coords[cell_min_x_index, cell_min_y_index, 0] = cell_min_x + cell_size/2
        grid_cells_coords[cell_min_x_index, cell_min_y_index, 1] = cell_min_y + cell_size/2


        for i_node in cell_node_indices:
            node_index = int(i_node)
            node_coords = h5_node_coords[node_index]
            # node_pressure = h5_node_pressure[node_index]
            node_displacement_x = h5_node_displacement_x[node_index]
            node_displacement_y = h5_node_displacement_y[node_index]
            node_x_coord = node_coords[0]
            node_y_coord = node_coords[1]
            node_x_index = int(np.round(node_x_coord/cell_size))
            node_y_index = int(np.round(node_y_coord/cell_size))

            if node_x_index < 0 or node_x_index >= nodes_per_x_dir or node_y_index < 0 or node_y_index >= nodes_per_y_dir:
                # raise error
                print(f'Node index out of bounds: {node_x_index}, {node_y_index}')
                Exception('Node index out of bounds')

            # Fill the grid nodes      
            grid_nodes_exist[node_x_index, node_y_index] = True
            grid_nodes_count[node_x_index, node_y_index] += 1
            # grid_nodes_response[node_x_index, node_y_index, 0] = node_pressure
            grid_nodes_response[node_x_index, node_y_index, 0] = node_displacement_x
            grid_nodes_response[node_x_index, node_y_index, 1] = node_displacement_y
            grid_nodes_coords[node_x_index, node_y_index, 0] = node_x_coord
            grid_nodes_coords[node_x_index, node_y_index, 1] = node_y_coord

    # print some stats
    if print_info:
        print(f'---------------------------------------------------------')
        print(f'Collected Information:')
        print(f'---------------------------------------------------------')
        print(f'Number of cells           : {np.sum(grid_cells_exist)}')
        print(f'Number of cells+duplicates: {np.sum(grid_cells_count)}')
        print(f'Number of nodes           : {np.sum(grid_nodes_exist)}')
        print(f'Number of nodes+duplicates: {np.sum(grid_nodes_count)}')
      
      

    # Mapping arrays
    map_nodes_grid_to_list = np.full((nodes_per_x_dir, nodes_per_y_dir), int(-1e5), dtype=int)
    map_cells_grid_to_list = np.full((cells_per_x_dir, cells_per_y_dir), int(-1e5), dtype=int)
    map_nodes_list_to_grid = np.full((n_unique_nodes,2), int(-1e5), dtype=int)
    map_cells_list_to_grid = np.full((n_unique_cells,2), int(-1e5), dtype=int)
  
    if compute_maps:
      # Fill the mapping arrays
      node_index = 0
      for i in range(nodes_per_x_dir):
          for j in range(nodes_per_y_dir):
              if grid_nodes_exist[i,j]:
                  map_nodes_grid_to_list[i,j] = node_index
                  map_nodes_list_to_grid[node_index,0] = i
                  map_nodes_list_to_grid[node_index,1] = j
                  node_index += 1
    
      cell_index = 0
      for i in range(cells_per_x_dir):
          for j in range(cells_per_y_dir):
              if grid_cells_exist[i,j]:
                  map_cells_grid_to_list[i,j] = cell_index
                  map_cells_list_to_grid[cell_index,0] = i
                  map_cells_list_to_grid[cell_index,1] = j
                  cell_index += 1    

    return (
      grid_nodes_coords,
      grid_nodes_response,
      grid_nodes_exist,
      grid_nodes_count,
      grid_cells_coords,
      grid_cells_response,
      grid_cells_exist,
      grid_cells_count,
      map_nodes_grid_to_list,
      map_cells_grid_to_list,
      map_nodes_list_to_grid,
      map_cells_list_to_grid
    )


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
def custom_get_data(file_name, time_index, compute_maps, print_info, pre_file_path):
    # target_load_index = load_case
    target_time_index = time_index
    target_time_index_zero_padded = f'{target_time_index:04}'
    file_name = file_name.replace('stxxx', target_time_index_zero_padded)
    file_path = f'{pre_file_path}/{file_name}'
    
    (
      grid_nodes_coords,
      grid_nodes_response,
      grid_nodes_exist,
      grid_nodes_count,
      grid_cells_coords,
      grid_cells_response,
      grid_cells_exist,
      grid_cells_count,
      map_nodes_grid_to_list,
      map_cells_grid_to_list,
      map_nodes_list_to_grid,
      map_cells_list_to_grid
    )= collect_structured_data_from_h5(file_path, compute_maps, print_info)
    
    return grid_nodes_response, grid_nodes_coords

   
def plot_data_for_error_array(error_data, model_name, to_dealii_link_step, to_deeponet_link_step, noise_scale, noise_method_index, plots_save_folder, data_type_name):
    n_time_steps = error_data.shape[0]
    time_step_values = error_data[:,1]
    strain_err_values = error_data[:,5]
    pressure_err_values = error_data[:,9]
    # print(time_step_values)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_step_values, strain_err_values, label='deal.ii -> Strain')
    ax.plot(time_step_values, pressure_err_values, label='DeepONet -> Pressure')
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