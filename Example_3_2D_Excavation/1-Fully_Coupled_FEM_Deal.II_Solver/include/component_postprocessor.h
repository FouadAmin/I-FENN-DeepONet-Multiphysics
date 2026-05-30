#ifndef COMPONENT_POSTPROCESSOR_H // Include guard to prevent multiple inclusions
#define COMPONENT_POSTPROCESSOR_H

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

namespace component_postprocessor_space
{
  using namespace dealii;

  template <int dim>
  class ComponentPostProcessor : public DataPostprocessor<dim>
  {
  public:
    std::string required_name;
    unsigned int i_component;
    double scalar_coefficient;

    ComponentPostProcessor(std::string required_name, unsigned int i_component, double scalar_coefficient)
    {
      this->required_name = required_name;
      this->i_component = i_component;
      this->scalar_coefficient = scalar_coefficient;
    }

    virtual ~ComponentPostProcessor() {}

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
      // AssertDimension(input_data.solution_values.size(), computed_quantities.size());

      for (unsigned int p = 0; p < input_data.solution_values.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(), 1);
        computed_quantities[p][0] = input_data.solution_values[p](i_component) * scalar_coefficient;
      }
    }

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {

      std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      return interpretation;
    }

    virtual std::vector<std::string> get_names() const override
    {
      std::vector<std::string> names;
      names.emplace_back(required_name);

      return names;
    }
  };

}

#endif