// deal.II:
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/physics/transformations.h>
#include <mpi.h>
#include <petscviewer.h>

// C++:
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <cmath>

// custom files
#include "../include/body_functions.h"
#include "../include/boundary_functions.h"
#include "../include/helper_functions.h"
#include "../include/mesh_generator.h"
#include "../include/point_history.h"
#include "../include/main_postprocessor.h"
#include "../include/gradient_postprocessor.h"
#include "../include/strain_postprocessor.h"
#include "../include/stress_calculator.h"
#include "../include/stress_postprocessor.h"
#include "../include/strain_stress_postprocessor.h"
#include "../include/io_functions.h"
#include "../include/double_vectors.h"

#include "cnpy.h" // for reading/saving numpy arrays


namespace Program010
{
  using namespace dealii;
  using namespace helper_functions_space;
  using namespace point_history_space;
  using namespace body_functions_space;
  using namespace boundary_functions_space;
  using namespace mesh_generator_space;
  using namespace main_postprocessor_space;
  using namespace gradient_postprocessor_space;
  using namespace strain_postprocessor_space;
  using namespace stress_postprocessor_space;
  using namespace stress_calculator_space;
  using namespace strain_stress_postprocessor_space;

  // Constant expressions
  constexpr bool use_mass = false;
  constexpr int dim_init = 3;
  constexpr int dim_case = 0; // 0=> 3d , 1=> plane stress , 2=> plane strain , 3=> general plane (use tensors like 3d)
  constexpr bool print_system_matrix = false;
  constexpr bool save_hdf5 = true;
  constexpr bool save_xdmf = true; // usefull for paraview in case of hdf5 output
  constexpr bool save_vtu = true;  // usefull for paraview time steps
  constexpr bool save_pvd = true;  // usefull for paraview all steps referenced in one file

  // ## TopLevel Class
  template <int dim>
  class TopLevel
  {
  public:
    TopLevel();
    ~TopLevel();
    void run();

  private:
    void determine_component_extractors();

    void create_and_prepare_grid();

    void handle_constraints();

    void handle_boundary_values();

    void setup_system();

    void initialize_state_vectors();

    void assemble_system();

    void solve_timestep();

    unsigned int solve_system_equations();

    void update_state_vectors();

    void output_results() const;

    void update_time_increment_parameters();

    void do_initial_timestep();

    void do_timestep(); // for rest of steps

    void initialize_quadrature_point_history(); // for first time step

    void update_quadrature_point_history(); // at end of each time step

    parallel::shared::Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    std::vector<PointHistory<dim>> quadrature_point_history;

    // Blocks
    const FEValuesExtractors::Vector u_extractor;
    const FEValuesExtractors::Scalar t_extractor;
    static const unsigned int n_blocks = 2;
    static const unsigned int n_components = dim + 1;
    static const unsigned int first_u_component = 0;
    static const unsigned int t_component = dim;
    std::vector<unsigned int> block_component;
    enum
    {
      u_block = 0,
      t_block = 1
    };

    // DOF data
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    std::vector<IndexSet> locally_owned_partitioning;
    std::vector<IndexSet> locally_relevant_partitioning;
    std::vector<IndexSet> locally_owned_dofs_per_process;
    std::vector<IndexSet> locally_relevant_dofs_per_process;
    std::vector<types::global_dof_index> dofs_per_block;
    std::vector<types::global_dof_index> element_indices_u;
    std::vector<types::global_dof_index> element_indices_t;

    const QGauss<dim> quad_formula_cell;
    const QGauss<dim - 1> quad_formula_face;

    // Matrices & Vectors
    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::Vector system_rhs;
    Vector<double> non_mpi_solution;
    Vector<double> non_mpi_solution_increment;
    Vector<double> state_vector;
    Vector<double> state_vector_d;
    Vector<double> state_vector_dd;

    // Parallel communication
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

    // Time & History
  public:
    bool print_initial_step = true;
    bool print_step_time_instead_of_number = true; // true is the default (in pvd file)
    double t_stage1 = 0;
    double t_stage2 = 0;
    double t_end = t_stage1 + t_stage2;
    double t_current = 0;
    double t_dt_stage1 = 0;
    double t_dt_stage2 = 0;
    double t_dt = 0;
    unsigned int print_every_n_step = 1;
    unsigned int t_step_no = 0;

    // Logarithmic time stepping
    bool t_use_logarithmic = false; // if true specify t_total_steps_no, t_start, t_end ==> t_dt will be adaptive
    bool t_include_end = false;     // to mimic the calculations in np.logspace, but in our code t_end will be reached anyway
    bool t_use_cumulative = true;   // false is preferred; true to mimic the calculations steps np.logspace=>np.diff=>np.cumsum
    unsigned int t_total_steps_no = 50;
    double t_start = 1e+1; // 1e+0; //

    // Stress-Strain Properties
    bool input_by_lamda_mu = true;                        // true for Lamé's parameters, false for E and nu
    const double const_scale = 1e-6;                      // 1e-6 To convert stress from N/m2 to mpa (N/mm2)
    const double material_input_1 = 40.0e9 * const_scale; // Lamé's first parameter  const_lambda
    const double material_input_2 = 27.0e9 * const_scale; // Shear Modulus const_mu

    StressCalculator<dim> stress_calc;
    SymmetricTensor<2, dim> thermal_expansion_tensor;

    const double rho_density = 2700;                    // 2700 kg/m3 material density
    const double c_capacity = 910 * const_scale * 2700; // 910 J/(kg.K) * 2700 (kg/m3) => 910*2700 J/(m3.K) // volumetric heat capacity or thermal mass density
    const double k_conductivity = 237 * const_scale;    // 237 W/(m·K)  // thermal conductivity
    const double thermal_expansion_coeff = 2.31 * 1e-5; // 2.31 * 1e-5 m/(m.K) // (participates in coupling ==> off diagonal blocks)
    const double t_ref = 293;                           // K             // reference temperature
    const double t_gamma = 0.5;                         // gamma NewMark method (second order integration)
    const double t_beta = 0.25;                         // beta NewMark method (second order integration)
    const double t_theta = 1.0;                         // theta method (first order integration)

    // Boundary IDs
    const unsigned int boundary_id_z1 = 1;
    const unsigned int boundary_id_z2 = 2;
    const unsigned int boundary_id_r1 = 3;
    const unsigned int boundary_id_r2 = 4;

    // Used for Solver
    double first_step_tolerance = 0;

  public:
    std::string solution_folder = "./solution/";
    std::string solution_file_name = "Model010_V001";
    StrainStressPostProcessorOutputOptions stress_strain_postprocessor_output_options = StrainStressPostProcessorOutputOptions::vectors;
    MainPostProcessorOutputOptions main_postprocessor_output_options = MainPostProcessorOutputOptions::vectors;
    GradientPostProcessorOutputOptions gradient_postprocessor_output_options = GradientPostProcessorOutputOptions::vectors;
    GradientPostProcessorOutputOptions flux_postprocessor_output_options = GradientPostProcessorOutputOptions::vectors;
    DoubleVector1D current_lc_gen_data;
    int load_case_index;
    int mesh_refinement;
  };

  template <int dim>
  TopLevel<dim>::~TopLevel()
  {
    dof_handler.clear();
  }

  template <int dim>
  TopLevel<dim>::TopLevel()
      : triangulation(MPI_COMM_WORLD),
        fe(FE_Q<dim>(1) ^ dim, FE_Q<dim>(1)),
        dof_handler(triangulation),
        u_extractor(0),
        t_extractor(dim),
        dofs_per_block(n_blocks),
        quad_formula_cell(fe.degree + 1),
        quad_formula_face(fe.degree + 1),
        mpi_communicator(MPI_COMM_WORLD),
        n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
        pcout(std::cout, this_mpi_process == 0)
  {
    determine_component_extractors();

    if (input_by_lamda_mu)
    {
      stress_calc.set_properties_by_lambda_mu(material_input_1, material_input_2, dim_case);
    }
    else
    {
      stress_calc.set_properties_by_E_nu(material_input_1, material_input_2, dim_case);
    }

    pcout << "E= " << stress_calc.const_E << " : "
          << "nu= " << stress_calc.const_nu << " : "
          << "lambda= " << stress_calc.const_lambda << " : "
          << "mu= " << stress_calc.const_mu << std::endl;

    thermal_expansion_tensor = get_thermal_expansion_tensor<dim>(thermal_expansion_coeff);
  }

  template <int dim>
  void TopLevel<dim>::run()
  {
    do_initial_timestep();

    while (t_current < t_end)
      do_timestep();

    pcout << "Done ................." << std::endl;
  }

  template <int dim>
  void TopLevel<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_t.clear();
    for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
    {
      const unsigned int k_group = fe.system_to_base_index(k).first.first;
      if (k_group == u_block)
      {
        element_indices_u.push_back(k);
      }
      else if (k_group == t_block)
      {
        element_indices_t.push_back(k);
      }
      else
      {
        Assert(k_group <= t_block, ExcInternalError());
      }
    }
  }

  template <int dim>
  void TopLevel<dim>::create_and_prepare_grid()
  {
    double inner_radius = 1.0;
    double outer_radius = 2.0;
    double cylinder_length = 1.0;
    unsigned int n_radial_cells = 8;
    unsigned int n_axial_cells = 1;
    unsigned int refinement_level = mesh_refinement;
    CustomMeshGenerator<dim>::generate_cylinder_2(triangulation, inner_radius, outer_radius, cylinder_length, n_radial_cells, n_axial_cells, refinement_level);

    initialize_quadrature_point_history();
  }

  template <int dim>
  void TopLevel<dim>::setup_system()
  {
    pcout << "Setting up system ..." << std::endl;

    block_component =
        std::vector<unsigned int>(n_components, u_block); // Displacement
    block_component[t_component] = t_block;               // Temperature
    dof_handler.distribute_dofs(fe);
    dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    // Detecting locally owned and relevant dofs
    locally_owned_dofs.clear();
    locally_owned_partitioning.clear();
    locally_owned_partitioning.reserve(n_blocks);
    locally_owned_dofs_per_process = DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    Assert(locally_owned_dofs_per_process.size() > this_mpi_process, ExcInternalError());
    locally_owned_dofs = locally_owned_dofs_per_process[this_mpi_process];

    locally_relevant_dofs.clear();
    locally_relevant_partitioning.clear();
    locally_relevant_partitioning.reserve(n_blocks);
    locally_relevant_dofs_per_process = DoFTools::locally_relevant_dofs_per_subdomain(dof_handler);
    Assert(locally_relevant_dofs_per_process.size() > this_mpi_process, ExcInternalError());
    locally_relevant_dofs = locally_relevant_dofs_per_process[this_mpi_process];

    for (unsigned int b = 0; b < n_blocks; ++b)
    {
      const types::global_dof_index idx_begin = std::accumulate(dofs_per_block.begin(),
                                                                std::next(dofs_per_block.begin(), b), 0);
      const types::global_dof_index idx_end = std::accumulate(dofs_per_block.begin(),
                                                              std::next(dofs_per_block.begin(), b + 1), 0);
      locally_owned_partitioning.push_back(locally_owned_dofs.get_view(idx_begin, idx_end));
      locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(idx_begin, idx_end));
    }

    // Print DOFs Summary
    pcout
        << "  Number of active cells: " << triangulation.n_active_cells()
        << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      pcout
          << (p == 0 ? ' ' : '+')
          << (GridTools::count_cells_with_subdomain_association(triangulation, p));
    pcout << ")" << std::endl;

    pcout
        << "  Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      pcout
          << (p == 0 ? ' ' : '+')
          << (DoFTools::count_dofs_with_subdomain_association(dof_handler, p));
    pcout << ")" << std::endl;
    pcout
        << "  Number of degrees of freedom per block: "
        << "[n_u, n_t] = ["
        << dofs_per_block[u_block] << ", "
        << dofs_per_block[t_block] << "]"
        << std::endl;

    // Constraints
    handle_constraints(); // Constrains are updated internally to consider hanging nodes

    DynamicSparsityPattern dynamic_sparsity_pattern(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern, constraints,
                                    /*keep constrained dofs*/ false);
    SparsityTools::distribute_sparsity_pattern(
        dynamic_sparsity_pattern, locally_owned_dofs, mpi_communicator,
        locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dynamic_sparsity_pattern, mpi_communicator);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    non_mpi_solution.reinit(dof_handler.n_dofs());
    non_mpi_solution_increment.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void TopLevel<dim>::initialize_state_vectors()
  {
    // State vectors
    state_vector.reinit(dof_handler.n_dofs());
    state_vector_d.reinit(dof_handler.n_dofs());
    state_vector_dd.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void TopLevel<dim>::handle_constraints()
  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
  }

  template <int dim>
  void TopLevel<dim>::handle_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;
    const FEValuesExtractors::Vector u_components(0);
    const FEValuesExtractors::Scalar x_component(0);
    const FEValuesExtractors::Scalar y_component(1);
    const FEValuesExtractors::Scalar z_component(dim - 1);
    const FEValuesExtractors::Scalar t_component(dim);

    VectorTools::interpolate_boundary_values(dof_handler, boundary_id_z1,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values);

    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, tmp, system_rhs, false);
  }

  template <int dim>
  void TopLevel<dim>::assemble_system()
  {
    // integration constants
    const double a1 = 1 / (t_beta * t_dt * t_dt);
    const double a2 = 1 / (t_beta * t_dt);
    const double a3 = (1 - 2 * t_beta) / (2 * t_beta);

    const double dt_theta = t_dt * t_theta;
    const double dt_1_theta = t_dt * (1 - t_theta);

    const double dt_theta_by_ref = dt_theta / t_ref;
    const double dt_1_theta_by_ref = dt_1_theta / t_ref;

    const double c_by_ref = c_capacity / t_ref;
    const double k_by_ref = k_conductivity / t_ref;

    system_rhs = 0;
    system_matrix = 0;

    FEValues<dim> fe_values(fe, quad_formula_cell,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe, quad_formula_face,
                                     update_values | update_normal_vectors |
                                         update_quadrature_points |
                                         update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points_cell = quad_formula_cell.size();
    const unsigned int n_q_points_face = quad_formula_face.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    BodyForcesProvider<dim> body_forces_provider(t_current, t_dt, t_step_no, current_lc_gen_data);
    BodyForcesProvider<dim> body_forces_provider_old(t_current - t_dt, t_dt, t_step_no - 1, current_lc_gen_data);
    std::vector<Vector<double>> body_force_values(n_q_points_cell, Vector<double>(dim + 1));
    std::vector<Vector<double>> body_force_values_old(n_q_points_cell, Vector<double>(dim + 1));

    BoundaryNeumannProvider<dim> boundary_neumann_provider(t_current, t_dt, t_step_no, current_lc_gen_data, t_end);
    BoundaryNeumannProvider<dim> boundary_neumann_provider_old(t_current - t_dt, t_dt, t_step_no - 1, current_lc_gen_data, t_end);
    std::vector<Vector<double>> boundary_neumann_values(n_q_points_face, Vector<double>(dim + 1));
    std::vector<Vector<double>> boundary_neumann_values_old(n_q_points_face, Vector<double>(dim + 1));

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);

        // ## Cell Matrix ##
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // const unsigned int i_component = fe.system_to_component_index(i).first;
          const unsigned int i_block = fe.system_to_base_index(i).first.first;
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // const unsigned int j_component = fe.system_to_component_index(j).first;
            const unsigned int j_block = fe.system_to_base_index(j).first.first;
            for (unsigned int q = 0; q < n_q_points_cell; ++q)
            {
              const double JxW_q = fe_values.JxW(q);
              if ((i_block == u_block) && (j_block == u_block))
              {
                const SymmetricTensor<2, dim>
                    eps_phi_i = fe_values[u_extractor].symmetric_gradient(i, q),
                    eps_phi_j = fe_values[u_extractor].symmetric_gradient(j, q);

                cell_matrix(i, j) += (eps_phi_i * stress_calc.get(eps_phi_j)) * JxW_q;
                if (use_mass)
                  cell_matrix(i, j) += a1 * rho_density * (fe_values[u_extractor].value(i, q) * fe_values[u_extractor].value(j, q)) * JxW_q;
              }
              else if ((i_block == t_block) && (j_block == t_block))
              {
                cell_matrix(i, j) += c_by_ref *
                                         (fe_values[t_extractor].value(i, q) * fe_values[t_extractor].value(j, q)) *
                                         JxW_q +
                                     dt_theta * k_by_ref *
                                         (fe_values[t_extractor].gradient(i, q) * fe_values[t_extractor].gradient(j, q)) *
                                         JxW_q;
              }
              else if ((i_block == u_block) && (j_block == t_block))
              {
                const SymmetricTensor<2, dim> eps_phi_i = fe_values[u_extractor].symmetric_gradient(i, q);
                const SymmetricTensor<2, dim> alpha_n_j = thermal_expansion_tensor * fe_values[t_extractor].value(j, q);
                double B_ij = (eps_phi_i * stress_calc.get(alpha_n_j)) * JxW_q;
                cell_matrix(i, j) += -B_ij;
                cell_matrix(j, i) += B_ij;
              }
              else
              {
                Assert((i_block <= t_block) && (j_block <= t_block), ExcInternalError());
              }
            }
          }
        }

        // ## Body Forces to cell_rhs ##
        const PointHistory<dim> *local_quadrature_points_data =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

        body_forces_provider.vector_value_list(fe_values.get_quadrature_points(), body_force_values);
        body_forces_provider_old.vector_value_list(fe_values.get_quadrature_points(), body_force_values_old);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int i_component = fe.system_to_component_index(i).first;
          const unsigned int i_block = fe.system_to_base_index(i).first.first;
          for (unsigned int q = 0; q < n_q_points_cell; ++q)
          {
            const double JxW_q = fe_values.JxW(q);
            if (i_block == u_block)
            {
              const SymmetricTensor<2, dim> eps_phi_i = fe_values[u_extractor].symmetric_gradient(i, q);
              const SymmetricTensor<2, dim> &q_strain = local_quadrature_points_data[q].strain;
              const SymmetricTensor<2, dim> alpha_n_i = thermal_expansion_tensor * local_quadrature_points_data[q].temperature;
              const double E_i_U = (eps_phi_i * stress_calc.get(q_strain)) * JxW_q;
              const double B_i_T = (eps_phi_i * stress_calc.get(alpha_n_i)) * JxW_q;
              const double body_force_i = (body_force_values[q](i_component) * fe_values[u_extractor].value(i, q)[i_component]) * JxW_q;
              cell_rhs(i) += body_force_i - E_i_U + B_i_T;
              // cell_rhs(i) += (body_force_values[q](i_component) * fe_values[u_extractor].value(i, q)[i_component]) * JxW_q;
              if (use_mass)
              {
                const double q_ud = local_quadrature_points_data[q].velocity[i_component];
                const double q_udd = local_quadrature_points_data[q].acceleration[i_component];
                const double acc_term = a2 * q_ud + a3 * q_udd;
                cell_rhs(i) += (fe_values[u_extractor].value(i, q)[i_component] * rho_density * acc_term) * JxW_q;
              }
            }
            else if (i_block == t_block)
            {
              const double K_i_T = t_dt * k_by_ref * (fe_values[t_extractor].gradient(i, q) * local_quadrature_points_data[q].temperature_grad) * JxW_q;
              const double scaled_body_force_i = dt_theta_by_ref * body_force_values[q](i_component) +
                                                 dt_1_theta_by_ref * body_force_values_old[q](i_component);
              const double integrated_body_force_i = (scaled_body_force_i * fe_values[t_extractor].value(i, q)) * JxW_q;
              cell_rhs(i) += (-K_i_T + integrated_body_force_i);
            }
            else
            {
              Assert(i_block <= t_block, ExcInternalError());
            }
          }
        }

        // ## Boundary Face Forces to cell_rhs ##
        for (const auto face_no : cell->face_indices())
        {
          const auto &face = cell->face(face_no);
          if (face->at_boundary()) 
          {
            fe_face_values.reinit(cell, face);

            boundary_neumann_provider.vector_value_list(fe_face_values.get_quadrature_points(), boundary_neumann_values);
            boundary_neumann_provider_old.vector_value_list(fe_face_values.get_quadrature_points(), boundary_neumann_values_old);

            // use dofs_per_cell even for faces
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int i_component = fe.system_to_component_index(i).first;
              const unsigned int i_block = fe.system_to_base_index(i).first.first;
              for (unsigned int q = 0; q < n_q_points_face; ++q)
              {
                const double JxW_qf = fe_face_values.JxW(q);
                if (i_block == u_block)
                {
                  // NOTE fe_face_values.normal_vector(q) can be used if needed
                  cell_rhs(i) += (boundary_neumann_values[q](i_component) * fe_face_values[u_extractor].value(i, q)[i_component]) * JxW_qf;
                }
                else if (i_block == t_block)
                {
                  const double scaled_flux_i = dt_theta_by_ref * boundary_neumann_values[q](i_component) +
                                               dt_1_theta_by_ref * boundary_neumann_values_old[q](i_component);
                  const double integrated_flux_i = (scaled_flux_i * fe_face_values[t_extractor].value(i, q)) * JxW_qf;
                  cell_rhs(i) += -integrated_flux_i;
                }
                else
                {
                  Assert(i_block <= t_block, ExcInternalError());
                }
              }
            }
          }
        }

        // Transfer the local contributions to the linear system into the global
        // objects.
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    // Now compress the vector and the system matrix:
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    if (print_system_matrix && this_mpi_process == 0 && t_step_no == 2)
    {
      // Create a PETSc petsc_viewer_1 that writes to a file
      pcout << " ### Current dt is  " << t_dt << std::endl;
      PetscViewer petsc_viewer_1;
      PetscViewerCreate(PETSC_COMM_WORLD, &petsc_viewer_1);
      PetscViewerSetType(petsc_viewer_1, PETSCVIEWERASCII);
      PetscViewerFileSetName(petsc_viewer_1, "stiffness_matrix_.txt");
      MatView(system_matrix, petsc_viewer_1); // Output the matrix
      PetscViewerDestroy(&petsc_viewer_1);    // Cleanup
    }

    // Apply boundary values
    handle_boundary_values();

    pcout << " ==> Norm of rhs is " << system_rhs.l2_norm();
  }

  template <int dim>
  void TopLevel<dim>::solve_timestep()
  {
    assemble_system();

    const unsigned int n_iterations = solve_system_equations();

    pcout << " converged in " << n_iterations << " iterations ";

    update_state_vectors();

    update_quadrature_point_history();
  }

  template <int dim>
  unsigned int TopLevel<dim>::solve_system_equations()
  {
    double tol = 1.0e-16 * system_rhs.l2_norm();
    SolverControl solver_control(dof_handler.n_dofs() * 100, tol);

    PETScWrappers::SolverBicgstab solver(solver_control);

    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
    solver.solve(system_matrix, tmp, system_rhs, preconditioner);
    constraints.distribute(tmp);
    non_mpi_solution_increment = tmp;
    non_mpi_solution += non_mpi_solution_increment;

    return solver_control.last_step();
  }

  template <int dim>
  void TopLevel<dim>::update_state_vectors()
  {
    const double v1 = (t_gamma) / (t_beta * t_dt);
    const double v2 = (t_gamma - t_beta) / (t_beta);
    const double v3 = (t_gamma - 2 * t_beta) * t_dt / (2 * t_beta);

    const double a1 = 1 / (t_beta * t_dt * t_dt);
    const double a2 = 1 / (t_beta * t_dt);
    const double a3 = (1 - 2 * t_beta) / (2 * t_beta);

    // State vectors
    unsigned int n_dofs = dof_handler.n_dofs();
    Vector<double> new_sv(n_dofs);
    Vector<double> new_sv_d(n_dofs);
    Vector<double> new_sv_dd(n_dofs);

    for (unsigned int i = 0; i < n_dofs; i++)
    {
      new_sv[i] = non_mpi_solution[i];
      new_sv_d[i] = v1 * (new_sv[i] - state_vector[i]) - v2 * state_vector_d[i] - v3 * state_vector_dd[i];
      new_sv_dd[i] = a1 * (new_sv[i] - state_vector[i]) - a2 * state_vector_d[i] - a3 * state_vector_dd[i];
    }

    state_vector = new_sv;
    state_vector_d = new_sv_d;
    state_vector_dd = new_sv_dd;
  }

  template <int dim>
  void TopLevel<dim>::output_results() const
  {
    // check if the current time is the time to print
    if (t_step_no % print_every_n_step != 0)
    {
      return;
    }

    // Printing main components
    DataOut<dim> data_out;

    // VTK Flags
    DataOutBase::VtkFlags flags;
    // [no_compression , best_speed , best_compression , default_compression , plain_text]
    flags.compression_level = DataOutBase::CompressionLevel::default_compression;
    // flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);

    // Printing main solution using postprocessor
    MainPostProcessor<dim> main_postprocessor({"displacement", "temperature"}, main_postprocessor_output_options);
    data_out.add_data_vector(non_mpi_solution, main_postprocessor);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Computing Central Strain Trace
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Vector<double> cell_central_strain_trace(triangulation.n_active_cells());
    for (auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        // Get strain directly from stored point history
        const PointHistory<dim> *local_quadrature_points_data =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
        const SymmetricTensor<2, dim> &q_midpoint_strain = local_quadrature_points_data[0].midpoint_strain;
        double strain_trace_q_midpoint = q_midpoint_strain[0][0] + q_midpoint_strain[1][1] + q_midpoint_strain[2][2];
        cell_central_strain_trace(cell->active_cell_index()) = strain_trace_q_midpoint;
      }
      else
      {
        cell_central_strain_trace(cell->active_cell_index()) = 15e+20; //  just a large number to catch the eye, if everything is correct, this should not be saved in the output file
      }
    }
    data_out.add_data_vector(cell_central_strain_trace, "cell_central_strain_trace");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // End of Computing Central Strain Trace
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    data_out.build_patches();

    // ############################################
    // Saving the output
    // ############################################

    double time_to_write = 0.0;
    std::string time_step_filename = "";
    if (print_step_time_instead_of_number)
    {
      time_to_write = Utilities::truncate_to_n_digits(t_current, 6);
    }
    else
    {
      time_to_write = t_step_no / print_every_n_step;
    }

    if (save_hdf5)
    {
      // set filter settings
      bool filter_duplicate_vertices = true;
      // filter_duplicate_vertices -> Whether or not to filter out duplicate vertices and associated values
      // Setting this value to true will drastically reduce the output data size but will result in an output
      // file that does not faithfully represent the actual data if the data corresponds to discontinuous fields
      bool xdmf_hdf5_output = true;
      // xdmf_hdf5_output -> Whether the XDMF output refers to HDF5 files. This affects how output is structured.
      DataOutBase::DataOutFilterFlags data_out_filter_flags(filter_duplicate_vertices, xdmf_hdf5_output);
      DataOutBase::DataOutFilter data_out_filter(data_out_filter_flags);
      data_out.write_filtered_data(data_out_filter);

      const std::string init_filename = solution_file_name + "_" + Utilities::int_to_string(t_step_no, 4);
      const std::string h5_filename = init_filename + ".h5";
      data_out.write_hdf5_parallel(data_out_filter, solution_folder + h5_filename, MPI_COMM_WORLD);

      if (save_xdmf || load_case_index < 10)
      {
        const std::string xdmf_filename = init_filename + ".xdmf";
        std::vector<XDMFEntry> xdmf_entries;
        xdmf_entries.push_back(data_out.create_xdmf_entry(data_out_filter, h5_filename, time_to_write, MPI_COMM_WORLD));
        data_out.write_xdmf_file(xdmf_entries, solution_folder + xdmf_filename, MPI_COMM_WORLD);

        time_step_filename = xdmf_filename;
      }
      else
      {
        time_step_filename = h5_filename;
      }
    }

    if (save_vtu || load_case_index < 10)
    {
      const std::string vtu_filename = solution_file_name + "_" + Utilities::int_to_string(t_step_no, 4) + ".vtu";
      data_out.write_vtu_in_parallel(solution_folder + vtu_filename, mpi_communicator);

      time_step_filename = vtu_filename;
    }

    if ((save_pvd || load_case_index < 10) && this_mpi_process == 0)
    {
      static std::vector<std::pair<double, std::string>> times_and_names;
      times_and_names.emplace_back(time_to_write, time_step_filename);
      std::ofstream pvd_output(solution_folder + solution_file_name + ".pvd");
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
    }
  }

  template <int dim>
  void TopLevel<dim>::do_initial_timestep()
  {
    create_and_prepare_grid();
    setup_system();
    initialize_state_vectors();
    if (print_initial_step)
    {
      output_results(); // save timestep 0
    }

    update_time_increment_parameters();

    pcout << "    Number of active cells:       "
          << triangulation.n_active_cells() << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      pcout << (p == 0 ? ' ' : '+')
            << (GridTools::count_cells_with_subdomain_association(triangulation,
                                                                  p));
    pcout << ')' << std::endl;
    pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
          << " (by partition:";
    for (unsigned int p = 0; p < n_mpi_processes; ++p)
      pcout << (p == 0 ? ' ' : '+')
            << (DoFTools::count_dofs_with_subdomain_association(dof_handler, p));
    pcout << ')' << std::endl;

    solve_timestep();

    output_results();

    pcout << std::endl;
  }

  template <int dim>
  void TopLevel<dim>::update_time_increment_parameters()
  {
    if (t_use_logarithmic)
    {
      int count_to_consider = t_include_end ? (t_total_steps_no - 1) : (t_total_steps_no);
      double last_time = t_current;
      double time_scaling_factor = std::pow(t_end / t_start, 1.0 / count_to_consider);
      if (t_use_cumulative)
      {
        t_current = t_start * std::pow(time_scaling_factor, t_step_no + 1) - t_start;
      }
      else
      {
        t_current = t_start * std::pow(time_scaling_factor, t_step_no);
      }
      t_dt = t_current - last_time;
      ++t_step_no;
    }
    else
    {
      if (t_current < t_stage1)
      {
        t_dt = t_dt_stage1;
      }
      else
      {
        t_dt = t_dt_stage2;
      }

      t_current += t_dt;
      ++t_step_no;
      if (t_current > t_end)
      {
        t_dt -= (t_current - t_end);
        t_current = t_end;
      }
    }

    pcout << "Timestep No " << t_step_no << " at time " << t_current
          << " With increment " << t_dt;
  }
  template <int dim>
  void TopLevel<dim>::do_timestep()
  {
    // Synchronize all processes and measure the start time
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    update_time_increment_parameters();

    solve_timestep();

    output_results();

    // Measure the end time and synchronize
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the local execution time
    auto local_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    // Reduce times to find the maximum execution time across all processes
    long long max_duration;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    // Print the maximum execution time
    if (this_mpi_process == 0)
    {
      const int ms_per_h = 1000 * 60 * 60;
      const int ms_per_m = 1000 * 60;
      const int ms_per_s = 1000;
      pcout << " execution time: "
            << floor(max_duration / ms_per_h) << ":"
            << floor((max_duration % ms_per_h) / ms_per_m) << ":"
            << floor((max_duration % ms_per_m) / ms_per_s) << "."
            << max_duration % ms_per_s;
    }

    pcout << std::endl;
  }

  template <int dim>
  void TopLevel<dim>::initialize_quadrature_point_history()
  {
    triangulation.clear_user_data();

    {
      std::vector<PointHistory<dim>> tmp;
      quadrature_point_history.swap(tmp); // free all memory
    }
    quadrature_point_history.resize(triangulation.n_locally_owned_active_cells() *
                                    quad_formula_cell.size());

    unsigned int history_index = 0;
    for (auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell->set_user_pointer(&quadrature_point_history[history_index]);
        history_index += quad_formula_cell.size();
      }

    Assert(history_index == quadrature_point_history.size(), ExcInternalError());
  }

  template <int dim>
  void TopLevel<dim>::update_quadrature_point_history()
  {
    FEValues<dim> fe_values(fe, quad_formula_cell,
                            update_values | update_gradients);

    FEValues<dim> fe_values_midpoint(fe, QMidpoint<dim>(),
                                     update_values | update_gradients);

    std::vector<Tensor<1, dim>> displacement_values(quad_formula_cell.size());
    std::vector<Tensor<1, dim>> velocity_values(quad_formula_cell.size());
    std::vector<Tensor<1, dim>> acceleration_values(quad_formula_cell.size());
    std::vector<Tensor<2, dim>> displacement_grads(quad_formula_cell.size());

    std::vector<Tensor<2, dim>> midpoint_displacement_grads(fe_values_midpoint.n_quadrature_points);

    std::vector<double> temperature_values(quad_formula_cell.size());
    std::vector<Tensor<1, dim>> temperature_grads(quad_formula_cell.size());

    for (auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        // pointer to the quadrature point history
        PointHistory<dim> *local_quadrature_points_history = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
        Assert(local_quadrature_points_history >= &quadrature_point_history.front(), ExcInternalError());
        Assert(local_quadrature_points_history <= &quadrature_point_history.back(), ExcInternalError());

        //  update fe values
        fe_values.reinit(cell);
        fe_values[u_extractor].get_function_values(state_vector, displacement_values);
        fe_values[u_extractor].get_function_values(state_vector_d, velocity_values);
        fe_values[u_extractor].get_function_values(state_vector_dd, acceleration_values);
        fe_values[u_extractor].get_function_gradients(state_vector, displacement_grads);

        fe_values[t_extractor].get_function_values(state_vector, temperature_values);
        fe_values[t_extractor].get_function_gradients(state_vector, temperature_grads);

        //  update fe values for midpoint
        fe_values_midpoint.reinit(cell);
        fe_values_midpoint[u_extractor].get_function_gradients(state_vector, midpoint_displacement_grads);
        const SymmetricTensor<2, dim> midpoint_strain = get_strain(midpoint_displacement_grads[0]);

        // Then loop over the quadrature points of this cell:
        for (unsigned int q = 0; q < quad_formula_cell.size(); ++q)
        {
          const SymmetricTensor<2, dim> new_strain = get_strain(displacement_grads[q]);

          const SymmetricTensor<2, dim> new_stress = stress_calc.get(new_strain);

          local_quadrature_points_history[q].displacement = displacement_values[q];
          local_quadrature_points_history[q].velocity = velocity_values[q];
          local_quadrature_points_history[q].acceleration = acceleration_values[q];

          local_quadrature_points_history[q].strain = new_strain;
          local_quadrature_points_history[q].stress = new_stress;

          local_quadrature_points_history[q].temperature = temperature_values[q];
          local_quadrature_points_history[q].temperature_grad = temperature_grads[q];

          local_quadrature_points_history[q].midpoint_strain = midpoint_strain;
        }
      }
  }

} // namespace Program010

void run_one_elastic_problem(int load_case_index, DoubleVector1D current_lc_gen_data, int mesh_refinement, double total_load_time, int load_increments, std::string arg_solution_folder)
{
  using namespace dealii;
  using namespace Program010;

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Synchronize all processes and measure the start time
  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::high_resolution_clock::now();

  // Run the elastic problem
  TopLevel<dim_init> elastic_problem;
  elastic_problem.current_lc_gen_data = current_lc_gen_data;
  elastic_problem.load_case_index = load_case_index;
  elastic_problem.mesh_refinement = mesh_refinement;
  elastic_problem.solution_folder = arg_solution_folder;
  elastic_problem.t_stage2 = total_load_time;
  elastic_problem.t_end = elastic_problem.t_stage1 + elastic_problem.t_stage2;
  elastic_problem.t_dt_stage2 = total_load_time / load_increments;

  elastic_problem.solution_file_name += "_load" + std::to_string(load_case_index);
  elastic_problem.solution_file_name += "_mesh" + std::to_string(mesh_refinement);
  elastic_problem.solution_file_name += "_t" + std::to_string((int)total_load_time);
  elastic_problem.solution_file_name += "_inc" + std::to_string(load_increments);
  elastic_problem.solution_file_name += "_dt" + std::to_string((int)elastic_problem.t_dt_stage2);

  try
  {
    elastic_problem.run();
  }
  catch (const std::exception &e)
  {
    if (world_rank == 0)
    {
      std::cerr << e.what() << '\n';

      // append the error to the file
      std::ofstream error_file("logs/temp_log_errors.log", std::ios::app);
      error_file << "Load Case Index: " << elastic_problem.load_case_index
                 << " Current Time Index: " << elastic_problem.t_step_no
                 << std::endl;
    }
  }

  // Measure the end time and synchronize
  MPI_Barrier(MPI_COMM_WORLD);
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the local execution time
  auto local_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  // Reduce times to find the maximum execution time across all processes
  long long max_duration;
  MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG_LONG, MPI_MAX, 0,
             MPI_COMM_WORLD);

  // Print the maximum execution time
  if (world_rank == 0)
  {
    const int ms_per_h = 1000 * 60 * 60;
    const int ms_per_m = 1000 * 60;
    const int ms_per_s = 1000;
    std::cout << "Maximum execution time across all processes: "
              << floor(max_duration / ms_per_h) << ":"
              << floor((max_duration % ms_per_h) / ms_per_m) << ":"
              << floor((max_duration % ms_per_m) / ms_per_s) << "."
              << max_duration % ms_per_s << std::endl;

    const auto end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    int n_mpi_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_processes);

    // append the time to the file
    std::ofstream time_file("logs/temp_log_times.log", std::ios::app);
    time_file << "Load Case Index: " << load_case_index
              << " Number of MPI processes: " << n_mpi_processes
              << " time: "
              << floor(max_duration / ms_per_h) << ":"
              << floor((max_duration % ms_per_h) / ms_per_m) << ":"
              << floor((max_duration % ms_per_m) / ms_per_s) << "."
              << max_duration % ms_per_s
              << " finished at: " << std::ctime(&end_time); // << std::endl;
  }
}

int main(int argc, char **argv)
{
  try
  {
    using namespace dealii;
    using namespace Program010;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Print the number of processes
    int n_mpi_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_processes);
    if (world_rank == 0)
    {
      std::cout << "Number of MPI processes: " << n_mpi_processes << std::endl;
    }

    int load_case_index = std::stoi(argv[1]);
    int mesh_refinement = std::stoi(argv[2]);
    double t_total_load = std::stod(argv[3]);
    int t_load_increments_no = std::stoi(argv[4]);
    std::string arg_solution_folder = argv[5];

    if (world_rank == 0)
    {
      std::cout << "load_case_index: " << load_case_index << std::endl;
      std::cout << "mesh_refinement: " << mesh_refinement << std::endl;
      std::cout << "t_total_load: " << t_total_load << std::endl;
      std::cout << "t_load_increments_no: " << t_load_increments_no << std::endl;
      std::cout << "arg_solution_folder: " << arg_solution_folder << std::endl;
    }

    // Check if the solution folder exists
    if (!std::filesystem::exists(arg_solution_folder))
    {
      // Create the solution folder
      std::filesystem::create_directory(arg_solution_folder);
      std::cout << "Solution folder created." << std::endl;
    }

    std::string load_file_path = "input_deal.ii/Model010_Tube_flux_gen_matrix_v001.npy";
    // Read the matrix
    DoubleVector2D all_lc_gen_array = read_npy_file_as_DoubleVector2D(load_file_path);

    // Select the load case generation array for current load_case_index
    DoubleVector1D load_gen_array;
    if (load_case_index == -1)
    {
      load_gen_array = all_lc_gen_array[0];
      // set standard values
      load_gen_array[0] = 3;
      load_gen_array[1] = 8;
      load_gen_array[2] = 16;
      load_gen_array[3] = 2;
      load_gen_array[4] = 8;
      load_gen_array[5] = 16;
      load_gen_array[6] = 2;
    }
    else if (load_case_index == -2)
    {
      load_gen_array = all_lc_gen_array[0];
      // set standard values
      load_gen_array[0] = 1;
      load_gen_array[1] = 8;
      load_gen_array[2] = 16;
      load_gen_array[3] = 2;
      load_gen_array[4] = 8;
      load_gen_array[5] = 16;
      load_gen_array[6] = 2;
    }
    else if (load_case_index == -3)
    {
      load_gen_array = all_lc_gen_array[0];
      // set standard values
      load_gen_array[0] = 3;
      load_gen_array[1] = 8;
      load_gen_array[2] = 16;
      load_gen_array[3] = 2;
      load_gen_array[4] = -8;
      load_gen_array[5] = 16;
      load_gen_array[6] = 2;
    }
    else if (load_case_index == -4)
    {
      load_gen_array = all_lc_gen_array[0];
      // set standard values
      load_gen_array[0] = 1;
      load_gen_array[1] = 8;
      load_gen_array[2] = 16;
      load_gen_array[3] = 2;
      load_gen_array[4] = -8;
      load_gen_array[5] = 16;
      load_gen_array[6] = 2;
    }
    else
    {
      load_gen_array = all_lc_gen_array[load_case_index];
    }

    if (world_rank == 0)
    {
      // Print the dimensions of the matrix
      std::cout << "all_lc_gen_array dimensions: ("
                << all_lc_gen_array.size() << ", "
                << all_lc_gen_array[0].size() << ")"
                << std::endl;

      std::cout << "load_gen_array dimensions: ("
                << load_gen_array.size() << ")"
                << std::endl;

      double omega_t_i = load_gen_array[0];
      double q_inner_0_i = load_gen_array[1];
      double q_inner_1_i = load_gen_array[2];
      double omega_inner_r_i = load_gen_array[3];
      double q_outer_0_i = load_gen_array[4];
      double q_outer_1_i = load_gen_array[5];
      double omega_outer_r_i = load_gen_array[6];

      std::cout << "omega_t_i: " << omega_t_i << std::endl;
      std::cout << "q_inner_0_i: " << q_inner_0_i << std::endl;
      std::cout << "q_inner_1_i: " << q_inner_1_i << std::endl;
      std::cout << "omega_inner_r_i: " << omega_inner_r_i << std::endl;
      std::cout << "q_outer_0_i: " << q_outer_0_i << std::endl;
      std::cout << "q_outer_1_i: " << q_outer_1_i << std::endl;
      std::cout << "omega_outer_r_i: " << omega_outer_r_i << std::endl;
    }

    run_one_elastic_problem(load_case_index, load_gen_array, mesh_refinement, t_total_load, t_load_increments_no, arg_solution_folder);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
