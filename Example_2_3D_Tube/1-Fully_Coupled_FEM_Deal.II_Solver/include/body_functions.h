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

#include "../include/double_vectors.h"

// C++:
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace body_functions_space
{
    using namespace dealii;

    // BodyForcesProvider
    template <int dim>
    class BodyForcesProvider : public Function<dim>
    {
    public:
        BodyForcesProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector1D current_lc_gen_data);

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override;

        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &value_list) const override;

    private:
        const double t_current;
        const double t_dt;
        const int t_step_no;
        DoubleVector1D current_lc_gen_data;
        bool current_load_values_set = false;
        int load_application_step;
    };

    template <int dim>
    BodyForcesProvider<dim>::BodyForcesProvider(const double t_current, const double t_dt, const int t_step_no, DoubleVector1D current_lc_gen_data)
        : Function<dim>(dim + 1), t_current(t_current), t_dt(t_dt), t_step_no(t_step_no), current_lc_gen_data(current_lc_gen_data)
    {
    }

    template <int dim>
    inline void BodyForcesProvider<dim>::vector_value(const Point<dim> & /*p*/, Vector<double> &values) const
    {
        AssertDimension(values.size(), dim + 1);
        
        values = 0;
        return;
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
