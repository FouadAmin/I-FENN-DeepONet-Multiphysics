// deal.II: Poroelasticity Solve for inc. response
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
#include "../include/strain_stress_postprocessor.h"
#include "../include/strain_postprocessor.h"
#include "../include/stress_postprocessor.h"
#include "../include/component_postprocessor.h"
#include "../include/io_functions.h"
#include "../include/double_vectors.h"

namespace ProgramPE001
{
  using namespace dealii;
  using namespace helper_functions_space;
  using namespace point_history_space;
  using namespace body_functions_space;
  using namespace boundary_functions_space;
  using namespace mesh_generator_space;
  using namespace main_postprocessor_space;
  using namespace gradient_postprocessor_space;
  using namespace stress_calculator_space;
  using namespace strain_stress_postprocessor_space;
  using namespace strain_postprocessor_space;
  using namespace stress_postprocessor_space;
  using namespace component_postprocessor_space;

  // Constant expressions
  constexpr int dim_init = 2;
  constexpr int dim_case = 2; // 0=> 3d , 1=> plane stress , 2=> plane strain , 3=> general plane (like 3d)
  constexpr bool print_system_matrix = false;
  constexpr bool save_hdf5 = true;
  constexpr bool save_xdmf = true;            // usefull for paraview in case of hdf5 output
  constexpr bool save_vtu = true;             // usefull for paraview time steps
  constexpr bool save_pvd = true;             // usefull for paraview all steps referenced in one file
  constexpr int save_all_load_case_index = 10; // any load case less than this will be saved

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
    const FEValuesExtractors::Scalar p_extractor;
    static const unsigned int n_blocks = 2;
    static const unsigned int n_components = dim + 1;
    static const unsigned int first_u_component = 0;
    static const unsigned int p_component = dim;
    std::vector<unsigned int> block_component;
    enum
    {
      u_block = 0,
      p_block = 1
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
    std::vector<types::global_dof_index> element_indices_p;

    const QGauss<dim> quad_formula_cell;
    const QGauss<dim - 1> quad_formula_face;

    // Matrices & Vectors
    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::Vector system_rhs;
    Vector<double> non_mpi_solution;
    Vector<double> non_mpi_solution_inc;
    Vector<double> state_vector;
    Vector<double> state_vector_d;
    Vector<double> state_vector_dd;

    // Parallel communication
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;

  public:
    // Time & History
    bool print_initial_step = true;
    bool print_step_time_instead_of_number = false; // true is the default
    double t_end = 0;                               // 1e4;
    double t_current = 0;
    double t_dt = 0;
    int print_every_n_step = 1;
    unsigned int t_step_no = 0;

    // logaritmic time stepping
    bool t_use_logarithmic = false; // if true specify t_total_steps_no, t_start, t_end ==> t_dt will be adaptive
    bool t_include_end = true;      // to mimic the calculations in np.logspace, but in our code t_end will be reached anyway
    bool t_use_cumulative = false;  // to mimic the calculations steps np.logspace=>np.diff=>np.cumsum
    unsigned int t_total_steps_no = 25;
    double t_start = 1e-3; // 1e-3;

    // Material properties
    const double const_E = 14.51588 * 1e6;                                                    // Elastic Modulus
    const double const_nu = 0.3;                                                              // Poisson's ratio
    const double const_mu = const_E / (2 * (1 + const_nu));                                   // Shear Modulus (G or mu)
    const double const_lambda = (const_nu * const_E) / ((1 + const_nu) * (1 - 2 * const_nu)); // Lamé's first parameter
    const double const_K = const_lambda + (2.0 / 3.0) * const_mu;                             // Bulk Modulus K = lambda + (2/3)*mu

    const double const_K_s = 55556 * 1e6; // N/m2 // Elastic Modulus (Bulk modulus of the soil grains)
    const double const_K_w = 2.2 * 1e9;   // N/m2 // Elastic Modulus (Bulk modulus of the water)

    bool input_by_lamda_mu = true;                // true for Lamé's parameters, false for E and nu
    const double material_input_1 = const_lambda; // Lamé's first parameter  const_lambda
    const double material_input_2 = const_mu;     // Shear Modulus const_mu

    const double alpha_biot_coeff = 1.0 - const_K / const_K_s; // biot`s coeff. for clays often ranges from 0.7 to 1.0

    const double ground_acceleration = 10.00;                      // 9.81; // m/s2
    const double rho_w = 1000.0;                                   // 1000 kg/m3 // 1.0 ton/m3 water density
    const double gamma_w = rho_w * ground_acceleration;            // N/m3 // weight per unit volume of water
    const double mu_w = 1e-3;                                      // 1.0e-3 N/m2.s = 10-3 Pa.s water dynamic viscosity
    const double k_intrinsic_permeability = 1e-4 * mu_w / gamma_w; // 3e-9;// m^2 (intrinsic permeability)

    // define gravity unit vector (per hydraulic head definition, it is in the opposite direction of the gravity vector)
    Tensor<1, dim> gravity_unit_vector;

    const double k_hydraulic_conductivity = gamma_w * k_intrinsic_permeability / mu_w; // m/s (hydraulic conductivity)

    const double k_permeability_coeff = k_intrinsic_permeability / mu_w; // m^2 / Pa.s (permeability coefficient)

    StressCalculator<dim> stress_calc;
    SymmetricTensor<4, dim> stress_strain_tensor;
    SymmetricTensor<2, dim> biot_tensor;
    SymmetricTensor<2, dim> intrinsic_permeability_tensor;
    SymmetricTensor<2, dim> hydraulic_conductivity_tensor;

    const double phi_porosity = 0.4;                                                                 
    const double const_S = (alpha_biot_coeff - phi_porosity) / const_K_s + phi_porosity / const_K_w; // Storage coefficient (1/Mb)

    // Time Integration constants
    const double t_gamma = 0.5; // gamma NewMark method (second order integration)
    const double t_beta = 0.25; // beta NewMark method (second order integration)
    const double t_theta = 1.0; // theta method (first order integration)

  public:
    std::string solution_folder_name = "./solution-tests/";
    std::string solution_file_name = "ExcavPE001_LoadV002";
    StrainStressPostProcessorOutputOptions stress_strain_postprocessor_output_options = StrainStressPostProcessorOutputOptions::vectors;
    MainPostProcessorOutputOptions main_postprocessor_output_options = MainPostProcessorOutputOptions::scalars;
    GradientPostProcessorOutputOptions gradient_postprocessor_output_options = GradientPostProcessorOutputOptions::vectors;
    GradientPostProcessorOutputOptions flux_postprocessor_output_options = GradientPostProcessorOutputOptions::vectors;
    DoubleVector2D current_lc_data;
    int load_case_index;
    int mesh_refinement;
  };

  template <int dim>
  TopLevel<dim>::TopLevel()
      : triangulation(MPI_COMM_WORLD),
        fe(FE_Q<dim>(2) ^ dim, FE_Q<dim>(1)),
        dof_handler(triangulation),
        u_extractor(0),
        p_extractor(dim),
        dofs_per_block(n_blocks),
        quad_formula_cell(fe.degree + 1), // quad_formula_cell(n) Generate a formula with n quadrature points (in each space direction), exact for polynomials of degree 2n-1.
        quad_formula_face(fe.degree + 1), // fe.degree Maximal polynomial degree of a shape function in a single coordinate direction.
        mpi_communicator(MPI_COMM_WORLD),
        n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
        pcout(std::cout, this_mpi_process == 0)
  {

    // Print FE system details
    pcout << "FE system: " << fe.get_name() << std::endl;
    pcout << "FE system degree: " << fe.degree << std::endl; // Maximal polynomial degree of a shape function in a single coordinate direction.
    pcout << "FE system components: " << fe.components << std::endl;
    pcout << "FE system dofs_per_cell: " << fe.dofs_per_cell << std::endl;
    pcout << "FE system dofs_per_vertex: " << fe.dofs_per_vertex << std::endl;
    pcout << "FE system dofs_per_quad: " << fe.dofs_per_quad << std::endl; // not including the degrees of freedom on the lines and vertices of the quadrilateral.
    pcout << "FE system dofs_per_line: " << fe.dofs_per_line << std::endl; // not including the degrees of freedom on the vertices of the line.

    determine_component_extractors();

    if (input_by_lamda_mu)
    {
      stress_calc.set_properties_by_lambda_mu(material_input_1, material_input_2, dim_case);
    }
    else
    {
      stress_calc.set_properties_by_E_nu(material_input_1, material_input_2, dim_case);
    }

    pcout
        << "K= " << const_K << " | "
        << "E= " << stress_calc.const_E << " | "
        << "nu= " << stress_calc.const_nu << " | "
        << "Lame`s 1st Const. (lambda)= " << stress_calc.const_lambda << " | "
        << "Shear Modulus (mu or G)= " << stress_calc.const_mu << std::endl;

    pcout << "alpha_biot_coeff = " << alpha_biot_coeff << " | "
          << "S = " << const_S << std::endl;

    pcout << "k_intrinsic_permeability [m2] = " << k_intrinsic_permeability << std::endl;
    pcout << "k_hydraulic_conductivity [m/s] = " << k_hydraulic_conductivity << std::endl;
    pcout << "k_permeability_coeff [m2/Pa.s] = " << k_permeability_coeff << std::endl;

    stress_strain_tensor = get_stress_strain_tensor<dim>(const_lambda, const_mu);
    biot_tensor = get_biot_tensor<dim>(alpha_biot_coeff);
    intrinsic_permeability_tensor = get_permeability_tensor<dim>(k_intrinsic_permeability);
    hydraulic_conductivity_tensor = get_permeability_tensor<dim>(k_hydraulic_conductivity);

    for (unsigned int i = 0; i < dim; ++i)
      gravity_unit_vector[i] = 0.0;
    gravity_unit_vector[1] = 1.0;
  }

  template <int dim>
  TopLevel<dim>::~TopLevel() { dof_handler.clear(); }

  template <int dim>
  void TopLevel<dim>::run()
  {
    do_initial_timestep();

    while ((t_current + 1e-6) < t_end)
      do_timestep();

    pcout << "Done ................." << std::endl;
  }

  template <int dim>
  void TopLevel<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_p.clear();
    for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
    {
      const unsigned int k_group = fe.system_to_base_index(k).first.first;
      if (k_group == u_block)
      {
        element_indices_u.push_back(k);
      }
      else if (k_group == p_block)
      {
        element_indices_p.push_back(k);
      }
      else
      {
        Assert(k_group <= p_block, ExcInternalError());
      }
    }
  }

  template <int dim>
  void TopLevel<dim>::create_and_prepare_grid()
  {
    CustomMeshGenerator<dim>::read_excavation_mesh_4(triangulation);

    if (mesh_refinement > 0)
    {
      triangulation.refine_global(mesh_refinement);
    }

    initialize_quadrature_point_history();
  }

  template <int dim>
  void TopLevel<dim>::setup_system()
  {
    pcout << "Setting up system ..." << std::endl;

    // Distribute and renumber DOFs
    block_component =
        std::vector<unsigned int>(n_components, u_block); // Displacement
    block_component[p_component] = p_block;               // Pressure
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
        << "[n_u, n_p] = ["
        << dofs_per_block[u_block] << ", "
        << dofs_per_block[p_block] << "]"
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
    non_mpi_solution_inc.reinit(dof_handler.n_dofs());
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
    const FEValuesExtractors::Scalar p_component(dim);

    VectorTools::interpolate_boundary_values(dof_handler, 5,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values, fe.component_mask(u_components));

    VectorTools::interpolate_boundary_values(dof_handler, 1,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values, fe.component_mask(x_component));
    VectorTools::interpolate_boundary_values(dof_handler, 2,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values, fe.component_mask(x_component));
    VectorTools::interpolate_boundary_values(dof_handler, 3,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values, fe.component_mask(x_component));
    VectorTools::interpolate_boundary_values(dof_handler, 4,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values, fe.component_mask(x_component));

    VectorTools::interpolate_boundary_values(dof_handler, 7,
                                             Functions::ZeroFunction<dim>(n_components),
                                             boundary_values, fe.component_mask(y_component));

    // apply boundary values for the first time step only
    if (t_step_no == 1)
    {
      VectorTools::interpolate_boundary_values(dof_handler, 6,
                                               BoundaryDirichletProvider<dim>(t_current, t_dt, t_step_no),
                                               boundary_values, fe.component_mask(p_component));
      VectorTools::interpolate_boundary_values(dof_handler, 1,
                                               BoundaryDirichletProvider<dim>(t_current, t_dt, t_step_no),
                                               boundary_values, fe.component_mask(p_component));
    }
    else
    {
      VectorTools::interpolate_boundary_values(dof_handler, 6,
                                               Functions::ConstantFunction<dim>(0.0, n_components),
                                               boundary_values, fe.component_mask(p_component));
      VectorTools::interpolate_boundary_values(dof_handler, 1,
                                               Functions::ConstantFunction<dim>(0.0, n_components),
                                               boundary_values, fe.component_mask(p_component));
    }

    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, tmp, system_rhs, false);
  }

  template <int dim>
  void TopLevel<dim>::assemble_system()
  {
    const double dt_theta = t_dt * t_theta;

    const double one_by_mu_w = 1 / mu_w;

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

    // Body and Traction values
    BodyForcesProvider<dim> body_forces_provider(t_current, t_dt, t_step_no);
    BodyForcesProvider<dim> body_forces_provider_old(t_current - t_dt, t_dt, t_step_no - 1);
    std::vector<Vector<double>> body_force_values(n_q_points_cell, Vector<double>(dim + 1));
    std::vector<Vector<double>> body_force_values_old(n_q_points_cell, Vector<double>(dim + 1));

    BoundaryNeumannProvider<dim> boundary_neumann_provider(t_current, t_dt, t_step_no, current_lc_data);
    BoundaryNeumannProvider<dim> boundary_neumann_provider_old(t_current - t_dt, t_dt, t_step_no - 1, current_lc_data);
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

                cell_matrix(i, j) += t_theta * (eps_phi_i * stress_calc.get(eps_phi_j)) * JxW_q;
              }
              else if ((i_block == p_block) && (j_block == p_block))
              {
                // S term + H term
                cell_matrix(i, j) += (fe_values[p_extractor].value(i, q) * const_S * fe_values[p_extractor].value(j, q)) *
                                         JxW_q +
                                     dt_theta * one_by_mu_w *
                                         (fe_values[p_extractor].gradient(i, q) * intrinsic_permeability_tensor * fe_values[p_extractor].gradient(j, q)) *
                                         JxW_q;
              }
              else if ((i_block == u_block) && (j_block == p_block))
              {
                const SymmetricTensor<2, dim> eps_phi_i = fe_values[u_extractor].symmetric_gradient(i, q);
                const SymmetricTensor<2, dim> alpha_n_j = biot_tensor * fe_values[p_extractor].value(j, q);
                double Q_ij = (eps_phi_i * alpha_n_j) * JxW_q;
                cell_matrix(i, j) += -t_theta * Q_ij;
                cell_matrix(j, i) += Q_ij;
              }
              else
              {
                Assert((i_block <= p_block) && (j_block <= p_block), ExcInternalError());
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
              const double f_term = (body_force_values[q](i_component) * fe_values[u_extractor].value(i, q)[i_component]) * JxW_q;

              const SymmetricTensor<2, dim> eps_phi_i = fe_values[u_extractor].symmetric_gradient(i, q);
              const SymmetricTensor<2, dim> &q_strain = local_quadrature_points_data[q].strain;
              const double u_term = -1.0 * (eps_phi_i * stress_calc.get(q_strain)) * JxW_q;

              const SymmetricTensor<2, dim> alpha_p_n_i = biot_tensor * local_quadrature_points_data[q].pressure;
              const double q_term = 1.0 * (eps_phi_i * alpha_p_n_i) * JxW_q;

              cell_rhs(i) += f_term + u_term + q_term;
            }
            else if (i_block == p_block)
            {
              // const double q_pressure = local_quadrature_points_data[q].pressure;
              const Tensor<1, dim> &q_pressure_grad = local_quadrature_points_data[q].pressure_grad;
              // const double s_term = (fe_values[p_extractor].value(i, q) * const_S * q_pressure) * JxW_q;
              const double h_term = t_dt * one_by_mu_w *
                                    (fe_values[p_extractor].gradient(i, q) * intrinsic_permeability_tensor * q_pressure_grad) * JxW_q;

              // gravity term
              const double g_term = t_dt * (fe_values[p_extractor].gradient(i, q) * hydraulic_conductivity_tensor * gravity_unit_vector) * JxW_q;

              // source/sink term
              const double f_term = t_dt * (body_force_values[q](i_component) * fe_values[p_extractor].value(i, q)) * JxW_q;

              cell_rhs(i) += (-h_term - g_term + f_term);
            }
            else
            {
              Assert(i_block <= p_block, ExcInternalError());
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
                  const double f_term = (boundary_neumann_values[q](i_component) * fe_face_values[u_extractor].value(i, q)[i_component]) * JxW_qf;
                  cell_rhs(i) += f_term;
                }
                else if (i_block == p_block)
                {
                  const double f_term = -t_dt * (boundary_neumann_values[q](i_component) * fe_face_values[p_extractor].value(i, q)) * JxW_qf;
                  cell_rhs(i) += f_term;
                }
                else
                {
                  Assert(i_block <= p_block, ExcInternalError());
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
  }

  template <int dim>
  void TopLevel<dim>::solve_timestep()
  {
    assemble_system();

    solve_system_equations();

    update_state_vectors();

    update_quadrature_point_history();
  }

  template <int dim>
  unsigned int TopLevel<dim>::solve_system_equations()
  {
    double rel_ratio = 0.0;
    double system_rhs_l2_norm = system_rhs.l2_norm();
    if (t_step_no == 1)
    {
      rel_ratio = 1.0e-16;
    }
    else
    {
      rel_ratio = 1.0e-8;
    }
    double tol_to_use = rel_ratio * system_rhs_l2_norm;

    pcout << std::scientific << std::setprecision(2)
          << " RHS Norm is " << system_rhs_l2_norm
          << " target ratio " << rel_ratio
          << " tol_to_use " << tol_to_use; // << std::endl removed to have convergence info in the same line

    SolverControl solver_control(dof_handler.n_dofs() * 1000, tol_to_use);

    PETScWrappers::SolverBicgstab solver(solver_control);                 // Best
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix); // Best

    PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

    // Try in a loop to solve the system
    unsigned int max_attempts = 4;
    for (unsigned int i_attempt = 0; i_attempt < max_attempts; ++i_attempt)
    {
      try
      {
        solver.solve(system_matrix, tmp, system_rhs, preconditioner);
        constraints.distribute(tmp);
        non_mpi_solution_inc = tmp;
        non_mpi_solution += non_mpi_solution_inc;

        int n_iterations = solver_control.last_step();
        double last_value = solver_control.last_value();
        double ratio = last_value / system_rhs_l2_norm;
        pcout << " iterations " << n_iterations
              << " last value " << last_value
              << " ratio " << ratio
              << " attempt " << i_attempt + 1 ; // << std::endl;

        return solver_control.last_step();
      }
      catch (const std::exception &e)
      {
        if (i_attempt == max_attempts)
        {
          pcout << "    Solver failed after " << i_attempt + 1 << " attempts." << std::endl;
          throw;
        }
      }
    }

    return 0;
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

    // #######################################################
    // ## Compute total head solution
    // #######################################################
    Vector<double> total_head_solution(dof_handler.n_dofs());

    FEValues<dim> fe_values(fe, quad_formula_cell,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        int n_dofs_per_cell = fe.n_dofs_per_cell();
        int n_vertices = GeometryInfo<dim>::vertices_per_cell;
        std::vector<types::global_dof_index> global_dof_indices(n_dofs_per_cell);
        cell->get_dof_indices(global_dof_indices);
        int target_component = 2; // 2 for pressure
        for (int i_vertex = 0; i_vertex < n_vertices; ++i_vertex)
        {
          // get global dof index at vertex
          int global_dof_index_at_vertex = cell->vertex_dof_index(i_vertex, target_component);
          Point<dim> vertex_point = cell->vertex(i_vertex);
          double elevation_value = vertex_point[1];
          double pressure_value = non_mpi_solution[global_dof_index_at_vertex];
          double total_head_value = elevation_value + pressure_value / gamma_w;
          total_head_solution[global_dof_index_at_vertex] = total_head_value;
        }
      }
    // #######################################################
    // ## End of computing total head solution
    // #######################################################

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Computing Central Strain Trace
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Vector<double> cell_central_strain_trace(triangulation.n_active_cells());
    // Special FEValues for QMidpoint
    FEValues<dim> fe_values_midpoint(fe, QMidpoint<dim>(),
                                     update_values | update_gradients |
                                         update_quadrature_points | update_JxW_values);
    std::vector<Tensor<2, dim>> displacement_grads_midpoint(1);

    // Loop over all the cells...
    for (auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        // Get strain directly from stored point history
        const PointHistory<dim> *local_quadrature_points_data =
            reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
        const SymmetricTensor<2, dim> &midpoint_strain = local_quadrature_points_data[0].midpoint_strain;

        double strain_trace_midpoint = 0;
        for (unsigned int d = 0; d < dim; ++d)
        {
          strain_trace_midpoint += midpoint_strain[d][d];
        }
        cell_central_strain_trace(cell->active_cell_index()) = strain_trace_midpoint;
      }
      else
      {
        cell_central_strain_trace(cell->active_cell_index()) = 15e+20; //  just a large number to catch the eye, if everything is correct, this should not be saved in the output file
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // End of Computing Central Strain Trace
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    DataOut<dim> data_out;
    // VTK Flags
    DataOutBase::VtkFlags flags;
    // [no_compression , best_speed , best_compression , default_compression , plain_text]
    flags.compression_level = DataOutBase::CompressionLevel::default_compression;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(cell_central_strain_trace, "cell_central_strain_trace", DataOut<dim>::type_cell_data);

    // Printing main solution using postprocessor
    MainPostProcessor<dim> main_postprocessor({"displacement", "pressure"}, main_postprocessor_output_options);
    data_out.add_data_vector(non_mpi_solution, main_postprocessor);

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

    if (save_hdf5 || load_case_index < save_all_load_case_index)
    {
      // set filter settings
      bool filter_duplicate_vertices = false;
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
      data_out.write_hdf5_parallel(data_out_filter, solution_folder_name + h5_filename, MPI_COMM_WORLD);

      if (save_xdmf || load_case_index < save_all_load_case_index)
      {
        const std::string xdmf_filename = init_filename + ".xdmf";
        std::vector<XDMFEntry> xdmf_entries;
        xdmf_entries.push_back(data_out.create_xdmf_entry(data_out_filter, h5_filename, time_to_write, MPI_COMM_WORLD));
        data_out.write_xdmf_file(xdmf_entries, solution_folder_name + xdmf_filename, MPI_COMM_WORLD);

        time_step_filename = xdmf_filename;
      }
      else
      {
        time_step_filename = h5_filename;
      }
    }

    if (save_vtu || load_case_index < save_all_load_case_index)
    {
      const std::string vtu_filename = solution_file_name + "_" + Utilities::int_to_string(t_step_no, 4) + ".vtu";
      data_out.write_vtu_in_parallel(solution_folder_name + vtu_filename, mpi_communicator);

      time_step_filename = vtu_filename;
    }

    if ((save_pvd || load_case_index < save_all_load_case_index) && this_mpi_process == 0)
    {

      static std::vector<std::pair<double, std::string>> times_and_names;
      times_and_names.emplace_back(time_to_write, time_step_filename);
      std::ofstream pvd_output(solution_folder_name + solution_file_name + ".pvd");
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
      t_current += t_dt;
      ++t_step_no;
      if (t_current > t_end)
      {
        t_dt -= (t_current - t_end);
        t_current = t_end;
      }
    }

    pcout << std::fixed << std::setprecision(10) << "Step# " << t_step_no
          << std::scientific << std::setprecision(3) << " @time " << t_current
          << " dt " << t_dt; // << std::endl;
  }
  template <int dim>
  void TopLevel<dim>::do_timestep()
  {

    update_time_increment_parameters();

    solve_timestep();

    output_results();

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

    std::vector<double> pressure_values(quad_formula_cell.size());
    std::vector<Tensor<1, dim>> pressure_grads(quad_formula_cell.size());

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

        fe_values[p_extractor].get_function_values(state_vector, pressure_values);
        fe_values[p_extractor].get_function_gradients(state_vector, pressure_grads);

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

          local_quadrature_points_history[q].pressure = pressure_values[q];
          local_quadrature_points_history[q].pressure_grad = pressure_grads[q];

          local_quadrature_points_history[q].midpoint_strain = midpoint_strain;
        }
      }
  }

} // namespace ProgramPE001

int main(int argc, char **argv)
{
  try
  {
    using namespace dealii;
    using namespace ProgramPE001;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Print the number of processes
    int n_mpi_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_processes);

    // read command line arguments
    int load_case_index = std::stoi(argv[1]);
    int mesh_refinement = std::stoi(argv[2]);
    double t_total_time = std::stod(argv[3]);
    int t_load_increments_no = std::stoi(argv[4]);
    std::string arg_solution_folder = argv[5];

    if (world_rank == 0)
    {
      std::cout << "Number of MPI processes: " << n_mpi_processes << std::endl;
      std::cout << "load_case_index: " << load_case_index << std::endl;
      std::cout << "mesh_refinement: " << mesh_refinement << std::endl;
      std::cout << "t_total_time: " << t_total_time << std::endl;
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

    // load input data
    std::string load_file_path = "input_deal.ii/generated_excav_flux_data_v002.npy";
    // Read the matrix
    DoubleVector3D all_lc_gen_array = read_npy_file_as_DoubleVector3D(load_file_path);
    if (world_rank == 0)
    {
      // Print the dimensions of the matrix
      std::cout << "all_lc_gen_array dimensions: ("
                << all_lc_gen_array.size() << ", "
                << all_lc_gen_array[0].size() << ", "
                << all_lc_gen_array[0][0].size() << ")" << std::endl;
    }

    DoubleVector2D current_lc_data = all_lc_gen_array[load_case_index];
    if (world_rank == 0)
    {
      // Print the dimensions of the matrix
      std::cout << "current_lc_data dimensions: ("
                << current_lc_data.size() << ", "
                << current_lc_data[0].size() << ")" << std::endl;
    }

    DoubleVector2D current_lc_data_interpolated = interpolate_DoubleVector2D_to_new_time_increment(current_lc_data, t_load_increments_no);
    if (world_rank == 0)
    {
      // Print the dimensions of the matrix
      std::cout << "current_lc_data_interpolated dimensions: ("
                << current_lc_data_interpolated.size() << ", "
                << current_lc_data_interpolated[0].size() << ")" << std::endl;
    }

    // ############################################
    //  RUN THE PROGRAM
    // ############################################

    // Synchronize all processes and measure the start time
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    TopLevel<dim_init> elastic_problem;
    elastic_problem.current_lc_data = current_lc_data_interpolated;
    elastic_problem.load_case_index = load_case_index;
    elastic_problem.mesh_refinement = mesh_refinement;
    elastic_problem.solution_folder_name = arg_solution_folder;
    elastic_problem.t_end = t_total_time;
    elastic_problem.t_dt = t_total_time / t_load_increments_no;

    elastic_problem.solution_file_name += "_load" + std::to_string(load_case_index);
    elastic_problem.solution_file_name += "_mesh" + std::to_string(mesh_refinement);
    elastic_problem.solution_file_name += "_t" + std::to_string((int)t_total_time);
    elastic_problem.solution_file_name += "_inc" + std::to_string(t_load_increments_no);
    elastic_problem.solution_file_name += "_dt" + std::to_string((int)elastic_problem.t_dt);

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
      const auto end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

      const int ms_per_h = 1000 * 60 * 60;
      const int ms_per_m = 1000 * 60;
      const int ms_per_s = 1000;
      std::cout << "Maximum execution time across all processes: "
                << std::fixed << std::setprecision(0)
                << floor(max_duration / ms_per_h) << ":"
                << floor((max_duration % ms_per_h) / ms_per_m) << ":"
                << floor((max_duration % ms_per_m) / ms_per_s) << " // "
                << " finished at: " << std::ctime(&end_time) << std::endl;

      std::cout << "Saved at: "
                << elastic_problem.solution_file_name << std::endl;

      // append the time to the file
      std::ofstream time_file("logs/temp_log_times.log", std::ios::app);
      time_file << "Name: " << elastic_problem.solution_file_name
                << " Load Case Index: " << load_case_index
                << " Number of MPI processes: " << n_mpi_processes
                << " time: "
                << floor(max_duration / ms_per_h) << ":"
                << floor((max_duration % ms_per_h) / ms_per_m) << ":"
                << floor((max_duration % ms_per_m) / ms_per_s) << "."
                << max_duration % ms_per_s
                << " finished at: " << std::ctime(&end_time); // << std::endl;
    }
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
