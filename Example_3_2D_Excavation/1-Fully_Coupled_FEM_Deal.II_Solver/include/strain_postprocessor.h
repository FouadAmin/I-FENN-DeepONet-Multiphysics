#ifndef STRAIN_POSTPROCESSOR_H // Include guard to prevent multiple inclusions
#define STRAIN_POSTPROCESSOR_H

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

namespace strain_postprocessor_space
{
  using namespace dealii;
  template <int dim>
  class StrainPostProcessor : public DataPostprocessorTensor<dim>
  {
  public:
    StrainPostProcessor(const std::string vector_name) : DataPostprocessorTensor<dim>(vector_name, update_gradients)
    {
    }

    virtual ~StrainPostProcessor() {}

    virtual void
    evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                          std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());

      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
      {
        AssertDimension(computed_quantities[p].size(),
                        (Tensor<2, dim>::n_independent_components));

        // solution_gradients[p][d][e] is the gradient of the solution at point [p] of the component [d] in the direction [e]
        for (unsigned int d = 0; d < dim; ++d)
          for (unsigned int e = 0; e < dim; ++e)
            computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))] = (input_data.solution_gradients[p][d][e] +
                                                                                                          input_data.solution_gradients[p][e][d]) /
                                                                                                         2;
      }
    }


    // No Need; Implemented automatically with DataPostprocessorTensor
    // virtual std::vector<
    //     DataComponentInterpretation::DataComponentInterpretation>
    // get_data_component_interpretation() const override
    // {
    //   // std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim * dim, DataComponentInterpretation::component_is_part_of_tensor);
    //   // return data_component_interpretation;

    //   // std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    //   // for (unsigned int i = 0; i < dim; ++i)
    //   //   for (unsigned int j = 0; j < dim; ++j)
    //   //     interpretation.push_back(DataComponentInterpretation::component_is_part_of_tensor);
    //   // return interpretation;
    // }

    // Does not affect the component names; only affects the main scalar/vector/tensor name
    // virtual std::vector<std::string> get_names() const override{
    //   std::vector<std::string> names;
    //   for (unsigned int i = 0; i < dim; ++i)
    //     for (unsigned int j = 0; j < dim; ++j)
    //       names.emplace_back("strain_" + std::to_string(i));
    //   return names;
    // }
  };

}

#endif