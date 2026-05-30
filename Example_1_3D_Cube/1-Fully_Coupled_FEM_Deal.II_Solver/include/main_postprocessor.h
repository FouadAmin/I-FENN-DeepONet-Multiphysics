#ifndef MAIN_POSTPROCESSOR_H // Include guard to prevent multiple inclusions
#define MAIN_POSTPROCESSOR_H

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

namespace main_postprocessor_space
{
  using namespace dealii;

  enum MainPostProcessorOutputOptions
  {
    // tensors = 1,
    vectors = 2,
    scalars = 3
  };

  template <int dim>
  class MainPostProcessor : public DataPostprocessor<dim>
  {
  public:
    MainPostProcessorOutputOptions output_option;
    std::vector<std::string> required_names;
    unsigned int n_extra_names;
    unsigned int n_names;
    MainPostProcessor(std::vector<std::string> required_names, MainPostProcessorOutputOptions output_option)
    {
      this->required_names = required_names;
      this->output_option = output_option;
      this->n_names = required_names.size();
      this->n_extra_names = required_names.size() - 1;
    }

    virtual ~MainPostProcessor() {}

    virtual UpdateFlags get_needed_update_flags() const override
    {
      // return update_values | update_gradients;
      // return update_gradients;
      return update_values;
    }

    virtual void
    evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                          std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_values.size(),
                      computed_quantities.size());

      for (unsigned int p = 0; p < input_data.solution_values.size(); ++p)
      {
        // computed_quantities[p].size() depends on the number of putput names and number of interpreter types added in:
        // get_names() and get_data_component_interpretation() functions
        // Assert is ignored in release mode
        // AssertDimension(computed_quantities[p].size(), (Tensor<2, dim>::n_independent_components));

        // Displacement Components
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[p][d] = input_data.solution_values[p](d);

        // Additional Components
        for (unsigned int d = 0; d < n_extra_names; ++d)
          computed_quantities[p][d + dim] = input_data.solution_values[p](d + dim);
      }
    }

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      // std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim * dim, DataComponentInterpretation::component_is_part_of_tensor);
      // return data_component_interpretation;

      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;

      // Strain and Stress
      switch (output_option)
      {
      case vectors:

        // Displacement Components
        for (unsigned int d = 0; d < dim; ++d)
          interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
        // Additional Components
        for (unsigned int d = 0; d < n_extra_names; ++d)
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        break;
      case scalars:

        // Displacement Components
        for (unsigned int d = 0; d < dim; ++d)
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        // Additional Components
        for (unsigned int d = 0; d < n_extra_names; ++d)
          interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        break;
      default:
        throw std::logic_error("Function not implemented 5");
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

        // Displacement Components
        for (unsigned int d = 0; d < dim; ++d)
          names.emplace_back(required_names[0]);
        // Additional Components
        for (unsigned int d = 1; d < n_names; ++d)
          names.emplace_back(required_names[d]);

        break;
      case scalars:
        // Displacement Components
        names.emplace_back(required_names[0] + "_x");
        names.emplace_back(required_names[0] + "_y");
        if (dim == 3)
          names.emplace_back(required_names[0] + "_z");
        // Additional Components
        for (unsigned int d = 1; d < n_names; ++d)
          names.emplace_back(required_names[d]);

        break;
      default:

        throw std::logic_error("Function not implemented 8");

        break;
      }

      return names;
    }
  };

}

#endif