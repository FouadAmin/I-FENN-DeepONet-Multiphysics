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
        BoundaryDirichletProvider(const double t_current, const double t_dt, const int t_step_no, const double total_time);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        const double total_time;
    };

    template <int dim>
    BoundaryDirichletProvider<dim>::BoundaryDirichletProvider(const double t_current, const double t_dt, const int t_step_no, const double total_time)
        : Function<dim>(dim), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), total_time(total_time)
    {
    }

    template <int dim>
    void
    BoundaryDirichletProvider<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim);

        double max_value = 10;
        double bc_application_time = 0.1*total_time;
        int bc_linear_steps = bc_application_time/t_dt;
        double bc_linear_step_value = max_value / bc_linear_steps;

        values = 0;
        if (t_step_no <= bc_linear_steps  && (std::fabs(p[1] - 1.0) < 1e-9)) // Y=1
        {
            values(1) = p[0] * bc_linear_step_value; // apply UY BC
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
        BoundaryNeumannProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D load_values);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        DoubleVector3D current_load_values;
        int load_application_step;
    };

    template <int dim>
    BoundaryNeumannProvider<dim>::BoundaryNeumannProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector4D load_values)
        : Function<dim>(dim), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no)
    {
        load_application_step = 1000;
        if (t_step_no >= load_application_step)
        {
            current_load_values = load_values[t_step_no - load_application_step];
        }
    }

    template <int dim>
    // void BoundaryNeumannProvider<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    void BoundaryNeumannProvider<dim>::vector_value(const Point<dim> & /*p*/, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim);

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
