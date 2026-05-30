#ifndef IO_FUNCTIONS_H // Include guard to prevent multiple inclusions
#define IO_FUNCTIONS_H

// deal.II:
#include <deal.II/base/exceptions.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/base/utilities.h>

#include "../include/double_vectors.h"
#include "cnpy.h" // for reading/saving numpy arrays

// C++:
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <cmath>

int copy_file(const std::string &src_file_path, const std::string &dst_file_path)
{
    std::ifstream src(src_file_path, std::ios::binary);
    std::ofstream dst(dst_file_path, std::ios::binary);
    dst << src.rdbuf();
    if (!src.is_open())
    {
        std::cerr << "Error: Could not open source file: " << src_file_path << std::endl;
        return 1;
    }
    if (!dst.is_open())
    {
        std::cerr << "Error: Could not open destination file: " << dst_file_path << std::endl;
        return 1;
    }
    if (dst.fail())
    {
        std::cerr << "Error: Writing to destination file failed." << std::endl;
        return 1;
    }

    src.close();
    dst.close();
    return 0;
}

void copy_code_files(std::string copying_solution_file_name, std::string program_file_name)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::string current_path_string = current_path.string();
    std::cout << "Current working directory: " << current_path_string << std::endl;

    std::string src_file_path;
    std::string dst_file_path;
    int copy_result;

    // create solution_sco folder if it does not exist
    std::filesystem::create_directory(current_path_string + "/solution_sco");

    src_file_path = current_path_string + "/source/" + program_file_name;
    dst_file_path = current_path_string + "/solution_sco/" + copying_solution_file_name + "_main.scocc";
    copy_result = copy_file(src_file_path, dst_file_path);
    if (copy_result == 0)
    {
        std::cout << "Scource Code Copied Successfully at: "
                  << copying_solution_file_name + "_main.scocc" << std::endl;
    }
    else
    {
        std::cerr << "Error: Copying Failed" << std::endl;
        return;
    }

    src_file_path = current_path_string + "/include/" + "boundary_functions.h";
    dst_file_path = current_path_string + "/solution_sco/" + copying_solution_file_name + "_bnd.scocc";
    copy_result = copy_file(src_file_path, dst_file_path);
    if (copy_result == 0)
    {
        std::cout << "Scource Code Copied Successfully at: "
                  << copying_solution_file_name + "_bnd.scocc" << std::endl;
    }
    else
    {
        std::cerr << "Error: Copying Failed" << std::endl;
        return;
    }

    src_file_path = current_path_string + "/include/" + "body_functions.h";
    dst_file_path = current_path_string + "/solution_sco/" + copying_solution_file_name + "_bod.scocc";
    copy_result = copy_file(src_file_path, dst_file_path);
    if (copy_result == 0)
    {
        std::cout << "Scource Code Copied Successfully at: "
                  << copying_solution_file_name + "_bod.scocc" << std::endl;
    }
    else
    {
        std::cerr << "Error: Copying Failed" << std::endl;
        return;
    }
}

// ############
// NPY Functions
// ############

DoubleVector2D read_npy_file_as_DoubleVector2D(const std::string &file_path)
{
    cnpy::NpyArray arr = cnpy::npy_load(file_path);
    double *loaded_data = arr.data<double>();

    DoubleVector2D matrix(arr.shape[0], DoubleVector1D(arr.shape[1]));

    for (size_t t = 0; t < arr.shape[0]; ++t)
    {
        for (size_t x = 0; x < arr.shape[1]; ++x)
        {
            matrix[t][x] = loaded_data[t * arr.shape[1] + x];
        }
    }

    return matrix;
}

DoubleVector4D read_npy_file_as_DoubleVector4D(const std::string &file_path)
{
    cnpy::NpyArray arr = cnpy::npy_load(file_path);
    double *loaded_data = arr.data<double>();

    // DoubleVector4D matrix(arr.shape[0], std::vector<std::vector<double>>(arr.shape[1], std::vector<double>(arr.shape[2])));
    // Load to 4D vector
    DoubleVector4D matrix(arr.shape[0], DoubleVector3D(arr.shape[1], DoubleVector2D(arr.shape[2], DoubleVector1D(arr.shape[3]))));

    for (size_t t = 0; t < arr.shape[0]; ++t)
    {
        for (size_t x = 0; x < arr.shape[1]; ++x)
        {
            for (size_t y = 0; y < arr.shape[2]; ++y)
            {
                for (size_t z = 0; z < arr.shape[3]; ++z)
                {
                    matrix[t][x][y][z] = loaded_data[t * arr.shape[1] * arr.shape[2] * arr.shape[3] + x * arr.shape[2] * arr.shape[3] + y * arr.shape[3] + z];
                }
            }
        }
    }

    return matrix;
}

DoubleVector2D interpolate_DoubleVector2D_to_new_time_increment(DoubleVector2D old_load_array, int new_increments)
{
    int old_time_steps = old_load_array.size();
    int old_increments = old_time_steps - 1;
    if (old_increments == new_increments)
    {
        return old_load_array;
    }

    int new_time_steps = new_increments + 1;
    int dim_x = old_load_array[0].size();

    DoubleVector2D new_load_array(new_time_steps, DoubleVector1D(dim_x, 0.0));

    // fill the values of the first time step
    for (int x = 0; x < dim_x; ++x)
    {
        new_load_array[0][x] = old_load_array[0][x];
    }

    // assume dt_old = 1.0
    double dt_old = 1.0;
    double dt_new = dt_old * old_increments / new_increments;
    // loop over the new time steps
    for (int i = 1; i < new_time_steps; ++i)
    {
        double time = i * dt_new;
        int old_time_index_lower = std::floor(time / dt_old);
        int old_time_index_upper = std::ceil(time / dt_old);
        double alpha = (time - old_time_index_lower * dt_old) / dt_old;

        for (int x = 0; x < dim_x; ++x)
        {
            new_load_array[i][x] = (1 - alpha) * old_load_array[old_time_index_lower][x] + alpha * old_load_array[old_time_index_upper][x];
        }
    }

    return new_load_array;
}

DoubleVector4D interpolate_DoubleVector4D_to_new_time_increment(DoubleVector4D old_load_array, int new_increments)
{
    int old_time_steps = old_load_array.size();
    int old_increments = old_time_steps - 1;
    if (old_increments == new_increments)
    {
        return old_load_array;
    }

    int new_time_steps = new_increments + 1;
    int dim_x = old_load_array[0].size();
    int dim_y = old_load_array[0][0].size();
    int dim_z = old_load_array[0][0][0].size();

    DoubleVector4D new_load_array(new_time_steps, DoubleVector3D(dim_x, DoubleVector2D(dim_y, DoubleVector1D(dim_z, 0.0))));

    // fill the values of the first time step
    for (int x = 0; x < dim_x; ++x)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int z = 0; z < dim_z; ++z)
            {
                new_load_array[0][x][y][z] = old_load_array[0][x][y][z];
            }
        }
    }

    // assume dt_old = 1.0
    double dt_old = 1.0;
    double dt_new = dt_old * old_increments / new_increments;
    // loop over the new time steps
    for (int i = 1; i < new_time_steps; ++i)
    {
        double time = i * dt_new;
        int old_time_index_lower = std::floor(time / dt_old);
        int old_time_index_upper = std::ceil(time / dt_old);
        double alpha = (time - old_time_index_lower * dt_old) / dt_old;

        for (int x = 0; x < dim_x; ++x)
        {
            for (int y = 0; y < dim_y; ++y)
            {
                for (int z = 0; z < dim_z; ++z)
                {
                    new_load_array[i][x][y][z] = (1 - alpha) * old_load_array[old_time_index_lower][x][y][z] + alpha * old_load_array[old_time_index_upper][x][y][z];
                }
            }
        }
    }

    return new_load_array;
}

void save_DoubleVector2D_as_npy(DoubleVector2D array, std::string file_name)
{
    long unsigned int time_steps = array.size();
    long unsigned int dim_x = array[0].size();

    std::vector<double> data(time_steps * dim_x);

    for (long unsigned int t = 0; t < time_steps; ++t)
    {
        for (long unsigned int x = 0; x < dim_x; ++x)
        {
            data[t * dim_x + x] = array[t][x];
        }
    }

    cnpy::npy_save(file_name, &data[0], {time_steps, dim_x}, "w");
}

void save_DoubleVector4D_as_npy(DoubleVector4D array, std::string file_name)
{
    long unsigned int time_steps = array.size();
    long unsigned int dim_x = array[0].size();
    long unsigned int dim_y = array[0][0].size();
    long unsigned int dim_z = array[0][0][0].size();

    std::vector<double> data(time_steps * dim_x * dim_y * dim_z);
    for (long unsigned int t = 0; t < time_steps; ++t)
    {
        for (long unsigned int x = 0; x < dim_x; ++x)
        {
            for (long unsigned int y = 0; y < dim_y; ++y)
            {
                for (long unsigned int z = 0; z < dim_z; ++z)
                {
                    data[t * dim_x * dim_y * dim_z + x * dim_y * dim_z + y * dim_z + z] = array[t][x][y][z];
                }
            }
        }
    }

    cnpy::npy_save(file_name, &data[0], {time_steps, dim_x, dim_y, dim_z}, "w");
}

// ############
// HDF5 Functions
// ############

DoubleVector4D read_h5_file_as_DoubleVector4D(const std::string &file_path)
{
    // Open the HDF5 file
    dealii::HDF5::File h5_file(file_path, dealii::HDF5::File::FileAccessMode::open);

    // Open the dataset named "data"
    auto dataset = h5_file.open_dataset("data");

    // Get the dimensions of the dataset
    std::vector<hsize_t> dims = dataset.get_dimensions();

    DoubleVector4D data(dims[0], DoubleVector3D(dims[1], DoubleVector2D(dims[2], DoubleVector1D(dims[3]))));

    // Read the data into a 1D vector
    std::vector<double> read_data = dataset.read<std::vector<double>>();

    // Fill the 4D vector with data from the 1D vector
    size_t index = 0;
    for (hsize_t i = 0; i < dims[0]; ++i)
    {
        for (hsize_t j = 0; j < dims[1]; ++j)
        {
            for (hsize_t k = 0; k < dims[2]; ++k)
            {
                for (hsize_t l = 0; l < dims[3]; ++l)
                {
                    data[i][j][k][l] = read_data[index++];
                }
            }
        }
    }

    return data;
}

void save_DoubleVector4D_as_h5(DoubleVector4D array, std::string file_name)
{
    // Get the dimensions of the 4D vector
    size_t time_steps = array.size();
    size_t dim_x = array[0].size();
    size_t dim_y = array[0][0].size();
    size_t dim_z = array[0][0][0].size();

    // Create a 1D vector to store the data
    std::vector<double> data(time_steps * dim_x * dim_y * dim_z);

    // Fill the 1D vector with data from the 4D vector
    size_t index = 0;
    for (size_t i = 0; i < time_steps; ++i)
    {
        for (size_t j = 0; j < dim_x; ++j)
        {
            for (size_t k = 0; k < dim_y; ++k)
            {
                for (size_t l = 0; l < dim_z; ++l)
                {
                    data[index++] = array[i][j][k][l];
                }
            }
        }
    }

    // Open the HDF5 file
    dealii::HDF5::File h5_file(file_name, dealii::HDF5::File::FileAccessMode::create);

    // Set the dimensions of the dataset
    std::vector<hsize_t> dimensions = {time_steps, dim_x, dim_y, dim_z};

    // Create a dataset named "data" in the HDF5 file
    auto dataset = h5_file.create_dataset<double>("data", dimensions);

    // Write data to the dataset
    dataset.write(data);
}
#endif // IO_FUNCTIONS_H