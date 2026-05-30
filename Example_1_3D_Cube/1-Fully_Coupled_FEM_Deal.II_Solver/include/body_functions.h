#ifndef BODY_FUNCTIONS_H // Include guard to prevent multiple inclusions
#define BODY_FUNCTIONS_H

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

// C++:
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace body_functions_space
{
    using namespace dealii;
    using DoubleVector3D = std::vector<std::vector<std::vector<double>>>;
    using DoubleVector4D = std::vector<std::vector<std::vector<std::vector<double>>>>;

    // BodyForcesProvider
    template <int dim>
    class BodyForcesProvider : public Function<dim>
    {
    public:
        BodyForcesProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D load_values);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        DoubleVector3D current_load_values;
        bool current_load_values_set = false;
        int load_application_step;
    };

    template <int dim>
    BodyForcesProvider<dim>::BodyForcesProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D load_values)
        : Function<dim>(dim + 1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no)
    {
        load_application_step = 1;
        if (t_step_no >= load_application_step)
        {
            current_load_values = load_values[t_step_no + 1 - load_application_step]; // +1 to skip the first stored load of zeros
            current_load_values_set = true;
        }
    }

    template <int dim>
    inline void BodyForcesProvider<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim + 1);

        values = 0;
        if (!current_load_values_set)
        {
            return;
        }

        double space_increment = 1.0 / 30.0;

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

        // check if the index is within the range
        if (x_index_1 >= current_load_values.size() || y_index_1 >= current_load_values[0].size() || z_index_1 >= current_load_values[0][0].size())
        {
            std::cout << "x_location: " << p[0] << ", y_location: " << p[1] << ", z_location: " << p[2] << std::endl;
            std::cout << "x_index_1: " << x_index_1 << ", y_index_1: " << y_index_1 << ", z_index_1: " << z_index_1 << std::endl;
            throw std::runtime_error("Index out of range");
        }

        // check if the index is within the range and if value is not nan
        if (std::isnan(current_load_values[x_index_1][y_index_1][z_index_1]))
        {
            std::cout << "x_index_1: " << x_index_1 << ", y_index_1: " << y_index_1 << ", z_index_1: " << z_index_1 << std::endl;
            throw std::runtime_error("Value is nan");
        }

        double load_scale = 2.0e-1;

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

        // Store the load value
        F[0] = current_load_values[x_index_0][y_index_0][z_index_0];
        F[1] = current_load_values[x_index_1][y_index_0][z_index_0];
        F[2] = current_load_values[x_index_1][y_index_1][z_index_0];
        F[3] = current_load_values[x_index_0][y_index_1][z_index_0];
        F[4] = current_load_values[x_index_0][y_index_0][z_index_1];
        F[5] = current_load_values[x_index_1][y_index_0][z_index_1];
        F[6] = current_load_values[x_index_1][y_index_1][z_index_1];
        F[7] = current_load_values[x_index_0][y_index_1][z_index_1];

        // Compute the load value
        double integrated_load = 0.0;
        for (unsigned int i = 0; i < 8; ++i)
        {
            integrated_load += N[i] * F[i];
        }
        integrated_load *= load_scale;
        values(dim) = integrated_load;
    }

    template <int dim>
    void BodyForcesProvider<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const
    {
        const unsigned int n_points = points.size();
        AssertDimension(value_list.size(), n_points);
        for (unsigned int p = 0; p < n_points; ++p)
            BodyForcesProvider<dim>::vector_value(points[p], value_list[p]);
    }

}

#endif // BODY_FORCES_H
