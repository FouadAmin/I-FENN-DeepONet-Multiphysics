#ifndef STRAINSTRESS_POSTPROCESSOR_H // Include guard to prevent multiple inclusions
#define STRAINSTRESS_POSTPROCESSOR_H

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

#include "../include/stress_calculator.h"

namespace strain_stress_postprocessor_space
{
  using namespace dealii;
  using namespace stress_calculator_space;

  enum StrainStressPostProcessorOutputOptions
  {
    tensors = 1,
    vectors = 2,
    scalars = 3
  };

  template <int dim>
  class StrainStressPostProcessor : public DataPostprocessor<dim>
  {
  public:
    StressCalculator<dim> stress_calculator;
    StrainStressPostProcessorOutputOptions output_option;
    StrainStressPostProcessor(StressCalculator<dim> stress_calculator, StrainStressPostProcessorOutputOptions output_option)
    {
      this->stress_calculator = stress_calculator;
      this->output_option = output_option;
    }

    virtual ~StrainStressPostProcessor() {}

    virtual UpdateFlags get_needed_update_flags() const override
    {
      // return update_values | update_gradients;
      // return update_values;
      return update_gradients;
    }

    virtual void
    evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                          std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());

      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        // computed_quantities[p].size() depends on the number of putput names and number of interpreter types added in:
        // get_names() and get_data_component_interpretation() functions
        // Assert is ignored in release mode
        // AssertDimension(computed_quantities[p].size(), (Tensor<2, dim>::n_independent_components));

        // Strain and Stress Calculation

        // solution_gradients[p][d][e] is the gradient of the solution at point [p] of the component [d] in the direction [e]
        SymmetricTensor<2, dim> strain_tensor;
        SymmetricTensor<2, dim> stress_tensor;

        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
          {
            strain_tensor[d][e] = (input_data.solution_gradients[p][d][e] +
                                   input_data.solution_gradients[p][e][d]) /
                                  2;
          }
        stress_tensor = stress_calculator.get(strain_tensor);

        unsigned int stress_shift;
        unsigned int trace_shift;
        // Output Options
        switch (output_option)
        {
        case tensors:
          stress_shift = dim * dim;
          trace_shift = 2 * dim * dim;
          for (unsigned int d = 0; d < dim; ++d)
          {
            for (unsigned int e = 0; e < dim; ++e)
            {
              computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))] = strain_tensor[d][e];
              computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e)) + stress_shift] = stress_tensor[d][e];
            }
          }

          break;
        case vectors:
        case scalars:
          if (dim == 2)
          {
            stress_shift = dim;
            trace_shift = 2 * (dim + 1);
            // strain ii
            for (unsigned int d = 0; d < dim; ++d)
              computed_quantities[p][d] = strain_tensor[d][d];
            // stress ii
            for (unsigned int d = 0; d < dim; ++d)
              computed_quantities[p][d + stress_shift] = stress_tensor[d][d];
            // strain ij xy
            computed_quantities[p][0 + 2 * stress_shift] = strain_tensor[0][1];
            // stress ij xy
            computed_quantities[p][1 + 2 * stress_shift] = stress_tensor[0][1];
          }
          else if (dim == 3)
          {
            stress_shift = dim;
            trace_shift = 4 * dim;
            // strain ii
            for (unsigned int d = 0; d < dim; ++d)
              computed_quantities[p][d] = strain_tensor[d][d];
            // stress ii
            for (unsigned int d = 0; d < dim; ++d)
              computed_quantities[p][d + stress_shift] = stress_tensor[d][d];
            // strain ij xy,yz,xz
            computed_quantities[p][0 + 2 * stress_shift] = strain_tensor[0][1];
            computed_quantities[p][1 + 2 * stress_shift] = strain_tensor[1][2];
            computed_quantities[p][2 + 2 * stress_shift] = strain_tensor[0][2];
            // stress ij xy,yz,xz
            computed_quantities[p][0 + 3 * stress_shift] = stress_tensor[0][1];
            computed_quantities[p][1 + 3 * stress_shift] = stress_tensor[1][2];
            computed_quantities[p][2 + 3 * stress_shift] = stress_tensor[0][2];
          }
          else
          {
            throw std::logic_error("Function not implemented 1");
          }
          break;
        default:

          throw std::logic_error("Function not implemented 2");

          break;
        }

        // strain_trace & stress_trace
        computed_quantities[p][trace_shift] = trace(strain_tensor);
        computed_quantities[p][trace_shift + 1] = trace(stress_tensor);
      }
    }

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      // std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(dim * dim, DataComponentInterpretation::component_is_part_of_tensor);
      // return interpretation;

      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;

      // Strain and Stress
      switch (output_option)
      {
      case tensors:
        for (unsigned int k = 0; k < 2 * dim * dim; ++k)
          interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
        break;
      case vectors:
        if (dim == 2) // (xx,yy,00), xy
        {
          for (unsigned int k = 0; k < dim; ++k)
            interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
          for (unsigned int k = 0; k < dim; ++k)
            interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        }
        else if (dim == 3) // (xx,yy,zz), (xy,yz,zx)
        {
          for (unsigned int k = 0; k < 4 * dim; ++k)
            interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
        }
        else
        {
          throw std::logic_error("Function not implemented 3");
        }

        break;
      case scalars:
        if (dim == 2)
        {
          for (unsigned int k = 0; k < 2 * (dim + 1); ++k)
            interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        }
        else if (dim == 3)
        {
          for (unsigned int k = 0; k < 4 * dim; ++k)
            interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        }
        else
        {
          throw std::logic_error("Function not implemented 4");
        }
        break;
      default:
        throw std::logic_error("Function not implemented 5");
        break;
      }

      // Strain Trace & Stress Trace
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      return interpretation;
    }

    virtual std::vector<std::string> get_names() const override
    {
      std::vector<std::string> names;

      // Strain and Stress
      switch (output_option)
      {
      case tensors:

        for (unsigned int k = 0; k < dim * dim; ++k)
          names.emplace_back("strain_tensor");
        for (unsigned int k = 0; k < dim * dim; ++k)
          names.emplace_back("stress_tensor");

        break;
      case vectors:
        if (dim == 2)
        {
          for (unsigned int k = 0; k < dim; ++k)
            names.emplace_back("strain_xx_yy_00");

          for (unsigned int k = 0; k < dim; ++k)
            names.emplace_back("stress_xx_yy_00");

          names.emplace_back("strain_xy");

          names.emplace_back("stress_xy");
        }
        else if (dim == 3)
        {
          for (unsigned int k = 0; k < dim; ++k)
            names.emplace_back("strain_xx_yy_zz");
          for (unsigned int k = 0; k < dim; ++k)
            names.emplace_back("stress_xx_yy_zz");
          for (unsigned int k = 0; k < dim; ++k)
            names.emplace_back("strain_xy_yz_xz");
          for (unsigned int k = 0; k < dim; ++k)
            names.emplace_back("stress_xy_yz_xz");
        }
        else
        {
          throw std::logic_error("Function not implemented 6");
        }

        break;
      case scalars:
        if (dim == 2)
        {
          names.emplace_back("strain_xx");
          names.emplace_back("strain_yy");

          names.emplace_back("stress_xx");
          names.emplace_back("stress_yy");

          names.emplace_back("strain_xy");

          names.emplace_back("stress_xy");
        }
        else if (dim == 3)
        {
          names.emplace_back("strain_xx");
          names.emplace_back("strain_yy");
          names.emplace_back("strain_zz");

          names.emplace_back("stress_xx");
          names.emplace_back("stress_yy");
          names.emplace_back("stress_zz");

          names.emplace_back("strain_xy");
          names.emplace_back("strain_yz");
          names.emplace_back("strain_xz");

          names.emplace_back("stress_xy");
          names.emplace_back("stress_yz");
          names.emplace_back("stress_xz");
        }
        else
        {
          throw std::logic_error("Function not implemented 7");
        }

        break;
      default:

        throw std::logic_error("Function not implemented 8");

        break;
      }

      names.emplace_back("strain_trace");
      names.emplace_back("stress_trace");

      return names;
    }
  };

}

#endif