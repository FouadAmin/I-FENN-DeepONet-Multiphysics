#ifndef STRESS_CALCULATOR_H // Include guard to prevent multiple inclusions
#define STRESS_CALCULATOR_H

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

#include "../include/helper_functions.h"

namespace stress_calculator_space
{
  using namespace dealii;
  using namespace helper_functions_space;

  template <int dim>
  class StressCalculator
  {
  public:
    SymmetricTensor<4, dim> stress_strain_tensor;
    int dim_case;
    double const_lambda;
    double const_mu;
    double const_E;
    double const_nu;

    // Default constructor for initialization in the main program
    StressCalculator()
    {
    }

    StressCalculator(double const_lambda, double const_mu, double const_E, double const_nu, int dim_case)
    {
      this->const_lambda = const_lambda;
      this->const_mu = const_mu;
      this->const_E = const_E;
      this->const_nu = const_nu;
      this->dim_case = dim_case;
      stress_strain_tensor = get_stress_strain_tensor<dim>(const_lambda, const_mu);
    }

    void set_properties(double const_lambda, double const_mu, double const_E, double const_nu, int dim_case)
    {
      this->const_lambda = const_lambda;
      this->const_mu = const_mu;
      this->const_E = const_E;
      this->const_nu = const_nu;
      this->dim_case = dim_case;
      stress_strain_tensor = get_stress_strain_tensor<dim>(const_lambda, const_mu);
    }

    void set_properties_by_lambda_mu(double const_lambda, double const_mu, int dim_case)
    {
      // Equations are provided in the following link: https://en.wikipedia.org/wiki/Bulk_modulus
      double const_E;
      double const_nu;

      // 0=> 3d
      // 2=> plane strain
      // 3=> 2d like 3d implementation (tensor multiplication using lame's first parameter and shear modulus)
      if (dim_case == 0 || dim_case == 2 || dim_case == 3)
      {
        const_E = const_mu * (3 * const_lambda + 2 * const_mu) / (const_lambda + const_mu); // Elastic Modulus
        const_nu = 0.5 * const_lambda / (const_lambda + const_mu);                          // Poisson's ratio
      }
      else if (dim_case == 1) // 1=> plane stress
      {
        const_E = 4.0 * const_mu * (const_lambda + const_mu) / (const_lambda + 2.0 * const_mu); // Elastic Modulus
        const_nu = const_lambda / (const_lambda + 2.0 * const_mu);                              // Poisson's ratio
      }
      else
      {
        throw std::logic_error("Function not implemented");
      }

      this->dim_case = dim_case;
      this->const_lambda = const_lambda;
      this->const_mu = const_mu;
      this->const_E = const_E;
      this->const_nu = const_nu;
      stress_strain_tensor = get_stress_strain_tensor<dim>(const_lambda, const_mu);
    }

    void set_properties_by_E_nu(double const_E, double const_nu, int dim_case)
    {
      // Equations are provided in the following link: https://en.wikipedia.org/wiki/Bulk_modulus
      double const_mu;
      double const_lambda;

      // 0=> 3d
      // 2=> plane strain
      // 3=> 2d like 3d implementation (tensor multiplication using lame's first parameter and shear modulus)
      if (dim_case == 0 || dim_case == 2 || dim_case == 3)
      {
        const_lambda = (const_nu * const_E) / ((1 + const_nu) * (1 - 2 * const_nu)); // Lamé's first parameter
        const_mu = const_E / (2 * (1 + const_nu));                                   // Shear Modulus
      }
      else if (dim_case == 1) // 1=> plane stress
      {
        const_lambda = (const_nu * const_E) / ((1 + const_nu) * (1 - const_nu)); // Lamé's first parameter
        const_mu = const_E / (2 * (1 + const_nu));                               // Shear Modulus
      }
      else
      {
        throw std::logic_error("Function not implemented");
      }

      this->dim_case = dim_case;
      this->const_lambda = const_lambda;
      this->const_mu = const_mu;
      this->const_E = const_E;
      this->const_nu = const_nu;
      stress_strain_tensor = get_stress_strain_tensor<dim>(const_lambda, const_mu);
    }

    SymmetricTensor<2, dim> get(const SymmetricTensor<2, dim> &strain_tensor) const
    {
      // NOTE: For dim_case 1 and 2, both options using (E and nu, lambda and mu) can be implemented.
      // as the constants are updated accordingly using set_properties functions

      if (dim_case == 0 || dim_case == 3) // 0=> 3d, 3=> 2d like 3d implementation (tensor multiplication using lame's first parameter and shear modulus)
      {
        return stress_strain_tensor * strain_tensor;
      }
      else if (dim_case == 1) // 1=> plane stress
      {
        SymmetricTensor<2, dim> tmp;

        // # Elastic Modulus and Poisson Ratio
        // const double a = const_E / (1 - const_nu * const_nu);
        // tmp[0][0] = a * (strain_tensor[0][0] + const_nu * strain_tensor[1][1]);
        // tmp[1][1] = a * (const_nu * strain_tensor[0][0] + strain_tensor[1][1]);
        // tmp[0][1] = a * (1 - const_nu) * strain_tensor[0][1];

        // # Lame Constants
        tmp[0][0] = (const_lambda + 2.0 * const_mu) * strain_tensor[0][0] + const_lambda * strain_tensor[1][1];
        tmp[1][1] = (const_lambda + 2.0 * const_mu) * strain_tensor[1][1] + const_lambda * strain_tensor[0][0];
        tmp[0][1] = 2.0 * const_mu * strain_tensor[0][1];

        tmp[1][0] = tmp[0][1]; // could be ignored because of using SymmetricTensor
        return tmp;
      }
      else if (dim_case == 2) // 2=> plane strain
      {
        SymmetricTensor<2, dim> tmp;

        // # Elastic Modulus and Poisson Ratio
        // const double a = const_E / ((1 + const_nu) * (1 - 2 * const_nu));
        // tmp[0][0] = a * ((1 - const_nu) * strain_tensor[0][0] + const_nu * strain_tensor[1][1]);
        // tmp[1][1] = a * (const_nu * strain_tensor[0][0] + (1 - const_nu) * strain_tensor[1][1]);
        // tmp[0][1] = a * (1 - 2 * const_nu) * strain_tensor[0][1];

        // # Lame Constants
        tmp[0][0] = (const_lambda + 2.0 * const_mu) * strain_tensor[0][0] + const_lambda * strain_tensor[1][1];
        tmp[1][1] = (const_lambda + 2.0 * const_mu) * strain_tensor[1][1] + const_lambda * strain_tensor[0][0];
        tmp[0][1] = 2.0 * const_mu * strain_tensor[0][1];

        tmp[1][0] = tmp[0][1]; // could be ignored because of using SymmetricTensor
        return tmp;
      }
      else
      {
        throw std::logic_error("Function not implemented");
      }
    }
  };

}

#endif