#ifndef PRESSURE_FUNCTIONS_H // Include guard to prevent multiple inclusions
#define PRESSURE_FUNCTIONS_H

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

namespace pressure_functions_space
{
    using namespace dealii;

    // QuadPressureProvider
    template <int dim>
    class QuadPressureProvider : public Function<dim>
    {
    public:
        QuadPressureProvider(const double t_current, const double t_dt, const int t_step_no, const double t_theta, const int mesh_refinement, DoubleVector3D pressure_values);

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;

        virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component = 0) const override;

        void value_list_custom(FESystem<dim> fe, FEValues<dim> &fe_values, const TriaActiveIterator<DoFCellAccessor<dim, dim, false>> &cell, const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component = 0) const;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        const double t_theta;
        const int mesh_refinement;
        DoubleVector2D to_use_pressure_values;
        bool to_use_pressure_values_set = false;
        int pressure_application_step;
    };

    template <int dim>
    QuadPressureProvider<dim>::QuadPressureProvider(const double t_current, const double t_dt, const int t_step_no, const double t_theta, const int mesh_refinement, DoubleVector3D pressure_values)
        : Function<dim>(1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), t_theta(t_theta), mesh_refinement(mesh_refinement)
    {
        pressure_application_step = 1;
        if (t_step_no >= pressure_application_step)
        {
            DoubleVector2D current_pressure_values = pressure_values[t_step_no + 1 - pressure_application_step]; // +1 to skip the first stored pressure of zeros
            DoubleVector2D previous_pressure_values = pressure_values[t_step_no - pressure_application_step];
            to_use_pressure_values = previous_pressure_values;
            // Loop through the 2D pressure values and add difference the difference time t_theta
            for (unsigned int i = 0; i < current_pressure_values.size(); ++i)
            {
                for (unsigned int j = 0; j < current_pressure_values[i].size(); ++j)
                {
                    to_use_pressure_values[i][j] += (current_pressure_values[i][j] - previous_pressure_values[i][j]) * t_theta;
                }
            }


            to_use_pressure_values_set = true;
        }
    }

    template <int dim>
    inline double QuadPressureProvider<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
    {
        // compute the number of increments passed
        if (!to_use_pressure_values_set)
        {
            return 0.0;
        }

        double cell_size_per_x = 1.0 * std::pow(2, mesh_refinement);
        double cell_size_per_y = 1.0 * std::pow(2, mesh_refinement);

        unsigned int x_index_0 = static_cast<unsigned int>(std::floor((p[0]) / cell_size_per_x));
        unsigned int y_index_0 = static_cast<unsigned int>(std::floor((p[1]) / cell_size_per_y));

        unsigned int x_index_1 = x_index_0 + 1;
        unsigned int y_index_1 = y_index_0 + 1;


        double x1 = x_index_0 * cell_size_per_x;
        double x2 = x_index_1 * cell_size_per_x;
        double y1 = y_index_0 * cell_size_per_y;
        double y4 = y_index_1 * cell_size_per_y;


        double x = p[0];
        double y = p[1];

        double xDiff = x2 - x1;
        double yDiff = y4 - y1;
        double area = xDiff * yDiff;

        // check if the index is within the range
        if (x_index_1 >= to_use_pressure_values.size() || y_index_1 >= to_use_pressure_values[0].size())
        {
            std::cout << "x_location: " << p[0] << ", y_location: " << p[1] << std::endl;
            std::cout << "x_index_1: " << x_index_1 << ", y_index_1: " << y_index_1 << std::endl;
            throw std::runtime_error("Index out of range");
        }

        // check if the index is within the range and if value is not nan
        if (std::isnan(to_use_pressure_values[x_index_1][y_index_1]))
        {
            std::cout << "x_index_1: " << x_index_1 << ", y_index_1: " << y_index_1 << std::endl;
            throw std::runtime_error("Value is nan");
        }

        // create vector of shape functions
        std::vector<double> N(4);
        std::vector<double> F(4);

        // Compute the shape functions
        N[0] = (x2 - x) * (y4 - y) / area;
        N[1] = (x - x1) * (y4 - y) / area;
        N[2] = (x - x1) * (y - y1) / area;
        N[3] = (x2 - x) * (y - y1) / area;

        // Store the pressure value
        F[0] = to_use_pressure_values[x_index_0][y_index_0];
        F[1] = to_use_pressure_values[x_index_1][y_index_0];
        F[2] = to_use_pressure_values[x_index_1][y_index_1];
        F[3] = to_use_pressure_values[x_index_0][y_index_1];

        // Compute the pressure value
        double integrated_pressure = 0.0;
        for (unsigned int i = 0; i < 4; ++i)
        {
            integrated_pressure += N[i] * F[i];
        }

        if (t_step_no == 10)
        {
            // print values to log file all N values and F values and integrated_pressure
            std::ofstream log_file("logs/log_file_pressure_shape_functions_1.log", std::ios::app);
            log_file << "time_index: " << t_step_no << " , x_index: " << x_index_0 << " , y_index: " << y_index_0
                     << " N[0]: " << N[0] << " , N[1]: " << N[1] << " , N[2]: " << N[2] << " , N[3]: " << N[3]
                     << " F[0]: " << F[0] << " , F[1]: " << F[1] << " , F[2]: " << F[2] << " , F[3]: " << F[3]
                     << " integrated_pressure: " << integrated_pressure << std::endl;
        }

        return integrated_pressure;
    }

    template <int dim>
    void QuadPressureProvider<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component) const
    {
        const unsigned int n_points = points.size();
        AssertDimension(values.size(), n_points);
        for (unsigned int p = 0; p < n_points; ++p)
            values[p] = QuadPressureProvider<dim>::value(points[p], component);
    }

    template <int dim>
    void QuadPressureProvider<dim>::value_list_custom(FESystem<dim> fe, FEValues<dim> &fe_values, const TriaActiveIterator<DoFCellAccessor<dim, dim, false>> &cell, const std::vector<Point<dim>> & /*points*/, std::vector<double> &values, const unsigned int component) const
    {
        fe_values.reinit(cell);
        // loop over cell vertices

        int n_vertices = GeometryInfo<dim>::vertices_per_cell;
        int n_dofs_per_cell = fe.n_dofs_per_cell();
        int n_required_values = values.size();

        // get current processor id
        int current_processor_id = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        if (current_processor_id == 0)
        {
            std::ofstream log_file("logs/local_nodes_and_vertices.log", std::ios::app);
            log_file << " n_vertices: " << n_vertices << " n_dofs_per_cell: " << n_dofs_per_cell << " n_required_values: " << n_required_values << std::endl;
        }

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
            double value_at_vertex = QuadPressureProvider<dim>::value(vertex_point, component);
            local_dof_values[local_dof_index_at_vertex] = value_at_vertex;
        }

        const FEValuesExtractors::Scalar scalar_extractor(target_component);
        fe_values[scalar_extractor].get_function_values_from_local_dof_values(local_dof_values, values);

    }

}

#endif // PRESSURE_FUNCTIONS_H
