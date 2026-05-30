#ifndef GRADIENT_POSTPROCESSOR_H // Include guard to prevent multiple inclusions
#define GRADIENT_POSTPROCESSOR_H

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

namespace gradient_postprocessor_space
{
  using namespace dealii;

  enum GradientPostProcessorOutputOptions
  {
    // tensors = 1,
    vectors = 2,
    scalars = 3
  };

  template <int dim>
  class GradientPostProcessor : public DataPostprocessor<dim>
  {
  public:
    unsigned int component_index;
    std::string vector_name;
    GradientPostProcessorOutputOptions output_option;
    double scalar_coefficient;
    GradientPostProcessor(const unsigned int component_index, const std::string vector_name)
    {
      this->component_index = component_index;
      this->vector_name = vector_name;
      this->output_option = GradientPostProcessorOutputOptions::vectors;
      this->scalar_coefficient = 1.0;
    }
    GradientPostProcessor(const unsigned int component_index, const std::string vector_name, GradientPostProcessorOutputOptions output_option, double scalar_coefficient)
    {
      this->component_index = component_index;
      this->vector_name = vector_name;
      this->output_option = output_option;
      this->scalar_coefficient = scalar_coefficient;
    }

    virtual ~GradientPostProcessor() {}

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
        AssertDimension(computed_quantities[p].size(), dim);
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[p][d] = input_data.solution_gradients[p][component_index][d] * scalar_coefficient;
      }
    }

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
      switch (output_option)
      {
      case vectors:
        interpretation = std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_part_of_vector);
        break;
      case scalars:
        interpretation = std::vector<DataComponentInterpretation::DataComponentInterpretation>(dim, DataComponentInterpretation::component_is_scalar);
        break;
      default:
        throw std::logic_error("Function not implemented in gradient postprocessor");
        break;
      }
      return interpretation;
    }

    virtual std::vector<std::string> get_names() const override
    {
      std::vector<std::string> names;
      switch (output_option)
      {
      case vectors:
        for (unsigned int k = 0; k < dim; ++k)
          names.emplace_back(vector_name);
        break;
      case scalars:
        names.emplace_back(vector_name + "_x");
        names.emplace_back(vector_name + "_y");
        if (dim == 3)
          names.emplace_back(vector_name + "_z");

        break;
      default:
        throw std::logic_error("Function not implemented in gradient postprocessor");
        break;
      }
      return names;
    }
  };

}

#endif