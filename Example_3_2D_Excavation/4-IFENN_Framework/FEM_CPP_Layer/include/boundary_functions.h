#ifndef BOUNDARY_FUNCTIONS_H // Include guard to prevent multiple inclusions
#define BOUNDARY_FUNCTIONS_H

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

namespace boundary_functions_space
{
    using namespace dealii;

    // --------------------------------------------
    // Class BoundaryDirichletProvider
    // ############################################

    template <int dim>
    class BoundaryDirichletProvider : public Function<dim>
    {
    public:
        BoundaryDirichletProvider(const double t_current, const double t_dt, const int t_step_no);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
    };

    template <int dim>
    BoundaryDirichletProvider<dim>::BoundaryDirichletProvider(const double t_current, const double t_dt, const int t_step_no)
        : Function<dim>(dim + 1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no)
    {
    }

    template <int dim>
    void
    BoundaryDirichletProvider<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim + 1);

        double p_left = 0.0;
        double p_right = 0.0;
        double h_left = 40.0;
        double h_right = 28.0;
        double distance = 60.0;
        double rho_g = 1000.0 * 10.0; // 1000 kg/m^3 * 10 m/s^2
        values = 0;
        if (std::fabs(p[0] - 0.0) < 1e-9)
        {
            values(dim) = (p_left + h_left - p[1]) * rho_g;
        }
        else if (std::fabs(p[0] - distance) < 1e-9)
        {
            values(dim) = (p_right + h_right - p[1]) * rho_g;
        }
        else if (std::fabs(p[1] - h_left) < 1e-9)
        {
            values(dim) = p_left * rho_g;
        }
        else if (std::fabs(p[1] - h_right) < 1e-9)
        {
            values(dim) = p_right * rho_g;
        }
    }

    template <int dim>
    void BoundaryDirichletProvider<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const
    {
        const unsigned int n_points = points.size();
        AssertDimension(value_list.size(), n_points);
        for (unsigned int p = 0; p < n_points; ++p)
            BoundaryDirichletProvider<dim>::vector_value(points[p], value_list[p]);
    }

    // --------------------------------------------
    // Class BoundaryNeumannProvider
    // ############################################

    template <int dim>
    class BoundaryNeumannProvider : public Function<dim>
    {
    public:
        BoundaryNeumannProvider(const double t_current, const double t_dt, const int t_step_no, const DoubleVector2D current_lc_data);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        const DoubleVector2D current_lc_data;
    };

    template <int dim>
    BoundaryNeumannProvider<dim>::BoundaryNeumannProvider(const double t_current, const double t_dt, const int t_step_no, const DoubleVector2D current_lc_data)
        : Function<dim>(dim + 1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), current_lc_data(current_lc_data)
    {
    }

    template <int dim>
    // void BoundaryNeumannProvider<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    void BoundaryNeumannProvider<dim>::vector_value(const Point<dim> & /*p*/, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim + 1);

        values = 0;
    }

    template <int dim>
    void BoundaryNeumannProvider<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const
    {
        const unsigned int n_points = points.size();
        AssertDimension(value_list.size(), n_points);
        for (unsigned int p = 0; p < n_points; ++p)
            BoundaryNeumannProvider<dim>::vector_value(points[p], value_list[p]);
    }

}

#endif // BOUNDARY_FORCES_H
