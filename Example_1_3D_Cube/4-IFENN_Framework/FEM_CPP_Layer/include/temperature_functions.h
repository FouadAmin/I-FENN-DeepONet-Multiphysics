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
        QuadTemperatureProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D temperature_values, int mesh_refinement);

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override;

        virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component = 0) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        DoubleVector3D current_temperature_values;
        bool current_temperature_values_set = false;
        int temperature_application_step;
        int mesh_refinement; // Mesh refinement (number of cells in each direction, e.g., 45 for 45x45x45 grid)
    };

    template <int dim>
    QuadTemperatureProvider<dim>::QuadTemperatureProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D temperature_values, int mesh_refinement)
        : Function<dim>(1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), mesh_refinement(mesh_refinement)
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

        double space_increment = 1.0 / mesh_refinement; // Assuming mesh_refinement is the number of cells in each direction

        unsigned int x_index_0 = static_cast<unsigned int>(std::floor((p[0]) / space_increment));
        unsigned int y_index_0 = static_cast<unsigned int>(std::floor((p[1]) / space_increment));
        unsigned int z_index_0 = static_cast<unsigned int>(std::floor((p[2]) / space_increment));

        unsigned int x_index_1 = x_index_0 + 1;
        unsigned int y_index_1 = y_index_0 + 1;
        unsigned int z_index_1 = z_index_0 + 1;

        double x1 = x_index_0 * space_increment;
        double x2 = x_index_1 * space_increment;
        double y1 = y_index_0 * space_increment;
        double y4 = y_index_1 * space_increment;
        double z1 = z_index_0 * space_increment;
        double z5 = z_index_1 * space_increment;

        double x = p[0];
        double y = p[1];
        double z = p[2];

        double xDiff = x2 - x1;
        double yDiff = y4 - y1;
        double zDiff = z5 - z1;
        double volume = xDiff * yDiff * zDiff;

        // create vector of shape functions
        std::vector<double> N(8);
        std::vector<double> F(8);

        // Compute the shape functions
        N[0] = (x2 - x) * (y4 - y) * (z5 - z) / volume;
        N[1] = (x - x1) * (y4 - y) * (z5 - z) / volume;
        N[2] = (x - x1) * (y - y1) * (z5 - z) / volume;
        N[3] = (x2 - x) * (y - y1) * (z5 - z) / volume;
        N[4] = (x2 - x) * (y4 - y) * (z - z1) / volume;
        N[5] = (x - x1) * (y4 - y) * (z - z1) / volume;
        N[6] = (x - x1) * (y - y1) * (z - z1) / volume;
        N[7] = (x2 - x) * (y - y1) * (z - z1) / volume;

        // Store the temperature value
        F[0] = current_temperature_values[x_index_0][y_index_0][z_index_0];
        F[1] = current_temperature_values[x_index_1][y_index_0][z_index_0];
        F[2] = current_temperature_values[x_index_1][y_index_1][z_index_0];
        F[3] = current_temperature_values[x_index_0][y_index_1][z_index_0];
        F[4] = current_temperature_values[x_index_0][y_index_0][z_index_1];
        F[5] = current_temperature_values[x_index_1][y_index_0][z_index_1];
        F[6] = current_temperature_values[x_index_1][y_index_1][z_index_1];
        F[7] = current_temperature_values[x_index_0][y_index_1][z_index_1];

        // Compute the temperature value
        double integrated_temperature = 0.0;
        for (unsigned int i = 0; i < 8; ++i)
        {
            integrated_temperature += N[i] * F[i];
        }

        return integrated_temperature;
    }

    template <int dim>
    void QuadTemperatureProvider<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int component) const
    {
        const unsigned int n_points = points.size();
        AssertDimension(values.size(), n_points);
        for (unsigned int p = 0; p < n_points; ++p)
            values[p] = QuadTemperatureProvider<dim>::value(points[p], component);
    }

}

#endif // TEMPERATURE_FUNCTIONS_H
