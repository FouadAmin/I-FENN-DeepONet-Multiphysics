#ifndef HELPER_FUNCTIONS_H // Include guard to prevent multiple inclusions
#define HELPER_FUNCTIONS_H

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

namespace helper_functions_space
{
    using namespace dealii;

    // symmetric strain of shape function
    template <int dim>
    inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                              const unsigned int shape_func,
                                              const unsigned int q_point)
    {
        // Declare a temporary that will hold the return value:
        SymmetricTensor<2, dim> tmp;

        // First, fill diagonal terms
        for (unsigned int i = 0; i < dim; ++i)
            tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

        // Then fill the rest of the strain tensor
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = i + 1; j < dim; ++j)
                tmp[i][j] =
                    (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                     fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                    2;

        return tmp;
    }

    // Symmetric strain tensor from the gradient of a vector-valued field.
    template <int dim>
    inline SymmetricTensor<2, dim>
    get_strain(const Tensor<2, dim> &grad)
    {
        SymmetricTensor<2, dim> strain;
        for (unsigned int i = 0; i < dim; ++i)
            strain[i][i] = grad[i][i];

        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = i + 1; j < dim; ++j)
                strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

        return strain;
    }

    template <int dim>
    SymmetricTensor<4, dim> get_stress_strain_tensor(const double lambda,
                                                     const double mu)
    {
        SymmetricTensor<4, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                                           ((i == l) && (j == k) ? mu : 0.0) +
                                           ((i == j) && (k == l) ? lambda : 0.0));
        return tmp;
    }

    template <int dim>
    SymmetricTensor<2, dim> get_biot_tensor(const double biot_coeff)
    {
        SymmetricTensor<2, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
        {
            for (unsigned int j = 0; j < dim; ++j)
            {
                if (i == j)
                {
                    tmp[i][j] = biot_coeff;
                }
                else
                {
                    tmp[i][j] = 0.0;
                }
            }
        }
        return tmp;
    }

    template <int dim>
    SymmetricTensor<2, dim> get_permeability_tensor(const double k_permeability)
    {
        SymmetricTensor<2, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
        {
            for (unsigned int j = 0; j < dim; ++j)
            {
                if (i == j)
                {
                    tmp[i][j] = k_permeability;
                }
                else
                {
                    tmp[i][j] = 0.0;
                }
            }
        }
        return tmp;
    }
    
    template <int dim>
    inline bool
    is_remainder_close(double remainder_value, double denomerator_value, double tolerance_value){
        if(std::abs(remainder_value)<tolerance_value)
        return true;
        if(std::abs(remainder_value-denomerator_value)<tolerance_value)
        return true;

        return false;
    }
    
    // A function that computes the rotation matrix (2d & 3d implementation)
    // 2d implementation:
    Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>> &grad_u);
    // 3d implementation:
    Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>> &grad_u);

}

#endif // HELPER_FUNCTIONS_H
