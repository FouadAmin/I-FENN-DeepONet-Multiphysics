#ifndef TEMPERATURE_FUNCTIONS_H // Include guard to prevent multiple inclusions
#define TEMPERATURE_FUNCTIONS_H

// deal.II:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/transformations.h>

#include "../include/double_vectors.h"
#include "../include/helper_functions.h"

// C++:
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace temperature_functions_space
{
    using namespace dealii;

    // QuadTemperatureProvider
    template <int dim>
    class QuadTemperatureProvider : public Function<dim>
    {
    public:
        QuadTemperatureProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D temperature_values);

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;

        virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component = 0) const override;

        void value_list_custom(FESystem<dim> fe, FEValues<dim> &fe_values, const TriaActiveIterator<DoFCellAccessor<dim, dim, false>> &cell, const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component = 0) const;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        DoubleVector3D current_temperature_values;
        bool current_temperature_values_set = false;
        int temperature_application_step;
    };

    template <int dim>
    QuadTemperatureProvider<dim>::QuadTemperatureProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D temperature_values)
        : Function<dim>(1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no)
    {
        temperature_application_step = 1;
        if (t_step_no >= temperature_application_step)
        {
            current_temperature_values = temperature_values[t_step_no + 1 - temperature_application_step]; // +1 to skip the first stored temperature of zeros
            current_temperature_values_set = true;
        }
    }

    template <int dim>
    inline double QuadTemperatureProvider<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
    {
        // compute the number of increments passed
        if (!current_temperature_values_set)
        {
            return 0.0;
        }

        double cell_size_per_a = (2.0 * M_PI) / 64.0;
        double cell_size_per_r = 1.0 / 8.0;
        double cell_size_per_z = 1.0 / 8.0;

        double point_x = p[0];
        double point_y = p[1];
        double point_z = p[2];
        double point_r = std::sqrt(point_x * point_x + point_y * point_y);
        double point_a = std::atan2(point_y, point_x);
        // if abs angle is close to 0, convert to 0
        if (std::abs(point_a) < (2 * M_PI * 1e-3))
            point_a = 0;
        // if angle is negative, convert to positive
        if (point_a < 0)
            point_a += 2 * M_PI;

        unsigned int a_index = (int)std::round(point_a / cell_size_per_a);
        unsigned int r_index = (int)std::round((point_r - 1) / cell_size_per_r);
        unsigned int z_index = (int)std::round(point_z / cell_size_per_z);
        // end of loop condition
        if (a_index == 64)
            a_index = 0;

        double a_remainder = std::fmod(point_a, cell_size_per_a);
        double r_remainder = std::fmod(point_r - 1, cell_size_per_r);
        double z_remainder = std::fmod(point_z, cell_size_per_z);

        bool a_remainder_close = helper_functions_space::is_remainder_close<dim>(a_remainder, cell_size_per_a, cell_size_per_a * 1e-3);
        bool r_remainder_close = helper_functions_space::is_remainder_close<dim>(r_remainder, cell_size_per_r, cell_size_per_r * 1e-3);
        bool z_remainder_close = helper_functions_space::is_remainder_close<dim>(z_remainder, cell_size_per_z, cell_size_per_z * 1e-3);

        bool is_vertex_on_cell = a_remainder_close && r_remainder_close && z_remainder_close;

        // raise error if vertex is not on cell
        if (!is_vertex_on_cell)
        {
            std::cout << "point_x: " << point_x << ", point_y: " << point_y << ", point_z: " << point_z << std::endl;
            std::cout << "point_a: " << point_a * 180 / M_PI << ", point_r: " << point_r << ", point_z: " << point_z << std::endl;
            std::cout << "a_index: " << a_index << ", r_index: " << r_index << ", z_index: " << z_index << std::endl;
            std::cout << "a_remainder: " << a_remainder * 180 / M_PI << ", r_remainder: " << r_remainder << ", z_remainder: " << z_remainder << std::endl;
            std::cout << "a_remainder_close: " << a_remainder_close << ", r_remainder_close: " << r_remainder_close << ", z_remainder_close: " << z_remainder_close << std::endl;
            throw std::runtime_error("Vertex is not on cell");
        }


        // check if the index is within the range
        if (a_index >= current_temperature_values.size() || r_index >= current_temperature_values[0].size() || z_index >= current_temperature_values[0][0].size())
        {
            std::cout << "x_location: " << p[0] << ", y_location: " << p[1] << ", z_location: " << p[2] << std::endl;
            std::cout << "a: " << point_a * 180 / M_PI << ", r: " << point_r << ", z: " << point_z << std::endl;
            std::cout << "a_index: " << a_index << ", r_index: " << r_index << ", z_index: " << z_index << std::endl;
            throw std::runtime_error("Index out of range");
        }

        // check if the index is within the range and if value is not nan
        if (std::isnan(current_temperature_values[a_index][r_index][z_index]))
        {
            std::cout << "a_index: " << a_index << ", r_index: " << r_index << ", z_index: " << z_index << std::endl;
            throw std::runtime_error("Value is nan");
        }

        // Compute the temperature value
        double stored_temperature = current_temperature_values[a_index][r_index][z_index];

        return stored_temperature;
    }

    template <int dim>
    void QuadTemperatureProvider<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component) const
    {
        const unsigned int n_points = points.size();
        AssertDimension(values.size(), n_points);
        for (unsigned int p = 0; p < n_points; ++p)
            values[p] = QuadTemperatureProvider<dim>::value(points[p], component);
    }

    template <int dim>
    void QuadTemperatureProvider<dim>::value_list_custom(FESystem<dim> fe, FEValues<dim> &fe_values, const TriaActiveIterator<DoFCellAccessor<dim, dim, false>> &cell, const std::vector<Point<dim>> &/*points*/, std::vector<double> &values, const unsigned int component) const
    {
        fe_values.reinit(cell);
        // loop over cell vertices

        int n_vertices = GeometryInfo<dim>::vertices_per_cell;
        int n_dofs_per_cell = fe.n_dofs_per_cell();

        std::vector<types::global_dof_index> global_dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(global_dof_indices);
        // create std::unordered_map<int, int> to store the global dof index and the local dof index
        std::unordered_map<int, int> global_to_local_dof_map;
        for (int i = 0; i < n_dofs_per_cell; ++i)
        {
            global_to_local_dof_map[global_dof_indices[i]] = i;
        }

        std::vector<double> local_dof_values(n_dofs_per_cell);
        int target_component = 0;
        for (int i_vertex = 0; i_vertex < n_vertices; ++i_vertex)
        {
            // get global dof index at vertex
            int global_dof_index_at_vertex = cell->vertex_dof_index(i_vertex, target_component);
            // get local dof index at vertex
            int local_dof_index_at_vertex = global_to_local_dof_map[global_dof_index_at_vertex];
            
            
            
            Point<dim> vertex_point = cell->vertex(i_vertex);
            double value_at_vertex = QuadTemperatureProvider<dim>::value(vertex_point, component);
            local_dof_values[local_dof_index_at_vertex] = value_at_vertex;
        }

        const FEValuesExtractors::Scalar scalar_extractor(target_component);
        fe_values[scalar_extractor].get_function_values_from_local_dof_values(local_dof_values, values);
    }

}

#endif // TEMPERATURE_FUNCTIONS_H
