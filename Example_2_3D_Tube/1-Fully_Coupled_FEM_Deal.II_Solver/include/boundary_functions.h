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
        : Function<dim>(dim + 1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), total_time(total_time)
    {
    }

    template <int dim>
    void
    BoundaryDirichletProvider<dim>::vector_value(const Point<dim> & /*p*/, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim + 1);

        values = 0;
        return;
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
        BoundaryNeumannProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector1D current_lc_gen_data, const double total_time);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        const double total_time;

        DoubleVector1D current_lc_gen_data;
    };

    template <int dim>
    BoundaryNeumannProvider<dim>::BoundaryNeumannProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector1D current_lc_gen_data, const double total_time)
        : Function<dim>(dim + 1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), total_time(total_time), current_lc_gen_data(current_lc_gen_data)
    {
    }

    template <int dim>
    void BoundaryNeumannProvider<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim + 1);

        values = 0;

        double inner_radius = 1.0;
        double outer_radius = 2.0;
        double cylinder_length = 1.0;
        double current_radius = std::sqrt(p[0] * p[0] + p[1] * p[1]);
        double radius_tolerance = 0.1; // allowance for nodes of faces to be considered on the boundary
        double theta = std::atan2(p[1], p[0]);


        // Temporal variation
        double temporal_frequency = current_lc_gen_data[0] * ((2 * M_PI) / total_time);
        double temporal_cos = std::cos(temporal_frequency * t_current);
        
        // Inner radial variation
        double inner_q0 = current_lc_gen_data[1]; // constant in time
        double inner_q1 = current_lc_gen_data[2]; // funtion of time
        double inner_radial_frequency = current_lc_gen_data[3];
        double inner_radial_cos = std::cos(inner_radial_frequency * theta);

        // Outer radial variation
        double outer_q0 = current_lc_gen_data[4];
        double outer_q1 = current_lc_gen_data[5];
        double outer_radial_frequency = current_lc_gen_data[6];
        double outer_radial_cos = std::cos(outer_radial_frequency * theta);

        if (p[2] > 0 && p[2] < cylinder_length) // within the cylinder length (on the side faces)
        {
            if ((std::fabs(current_radius - inner_radius) < radius_tolerance))
            {
                values(dim) = (-237e-6) * (inner_q0 + inner_q1 * temporal_cos * inner_radial_cos);
            }
            else if ((std::fabs(current_radius - outer_radius) < radius_tolerance))
            {
                values(dim) = (-237e-6) * (outer_q0 + outer_q1 * temporal_cos * outer_radial_cos);
            }
        }
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
