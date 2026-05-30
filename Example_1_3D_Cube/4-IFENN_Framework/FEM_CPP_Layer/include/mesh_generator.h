#ifndef MESH_GENERATOR_H // Include guard to prevent multiple inclusions
#define MESH_GENERATOR_H

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
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
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

namespace mesh_generator_space
{
    using namespace dealii;

    template <int dim>
    class CustomMeshGenerator
    {
    public:
        // generate testing cylinder
        // In a second step, we have to associated boundary conditions with the
        // upper and lower faces of the cylinder. We choose a boundary indicator of
        // 0 for the boundary faces that are characterized by their midpoints having
        // z-coordinates of either 0 (bottom face), 1 for (top face), 2 for all
        // faces on the inside of the cylinder shell, 3 for the outside.
        static void generate_cylinder_1(Triangulation<dim> &triangulation)
        {
            const double inner_radius = 790.0, outer_radius = 1000.0;
            const double cylinder_length = 4000.0;
            GridGenerator::cylinder_shell(triangulation,
                                          cylinder_length,
                                          inner_radius,
                                          outer_radius);
            for (const auto &cell : triangulation.active_cell_iterators())
                for (const auto &face : cell->face_iterators())
                    if (face->at_boundary())
                    {
                        const Point<dim> face_center = face->center();

                        if (std::fabs(face_center[2] - 0) < 1e-9)
                            face->set_boundary_id(0);
                        else if (std::fabs(face_center[2] - cylinder_length) < 1e-9)
                            face->set_boundary_id(1);
                        else if (std::sqrt(face_center[0] * face_center[0] +
                                           face_center[1] * face_center[1]) <
                                 (inner_radius + outer_radius) / 2)
                            face->set_boundary_id(2);
                        else
                            face->set_boundary_id(3);
                    }

            triangulation.refine_global(1);
            return;
        }

        static void generate_column_1(Triangulation<dim> &triangulation)
        {
            double width = 400.0;
            double height = 400.0;
            double total_length = height * 10;

            Point<dim> pointA_0(0, 0, height * 0);
            Point<dim> pointA_1(0, 0, height * 1);
            Point<dim> pointA_2(0, 0, height * 2);
            Point<dim> pointA_3(0, 0, height * 3);
            Point<dim> pointA_4(0, 0, height * 4);
            Point<dim> pointA_5(0, 0, height * 5);
            Point<dim> pointA_6(0, 0, height * 6);
            Point<dim> pointA_7(0, 0, height * 7);
            Point<dim> pointA_8(0, 0, height * 8);
            Point<dim> pointA_9(0, 0, height * 9);
            Point<dim> pointB_0(width, width, height * (1 + 0));
            Point<dim> pointB_1(width, width, height * (1 + 1));
            Point<dim> pointB_2(width, width, height * (1 + 2));
            Point<dim> pointB_3(width, width, height * (1 + 3));
            Point<dim> pointB_4(width, width, height * (1 + 4));
            Point<dim> pointB_5(width, width, height * (1 + 5));
            Point<dim> pointB_6(width, width, height * (1 + 6));
            Point<dim> pointB_7(width, width, height * (1 + 7));
            Point<dim> pointB_8(width, width, height * (1 + 8));
            Point<dim> pointB_9(width, width, height * (1 + 9));
            Triangulation<dim> new_cube_0;
            Triangulation<dim> new_cube_1;
            Triangulation<dim> new_cube_2;
            Triangulation<dim> new_cube_3;
            Triangulation<dim> new_cube_4;
            Triangulation<dim> new_cube_5;
            Triangulation<dim> new_cube_6;
            Triangulation<dim> new_cube_7;
            Triangulation<dim> new_cube_8;
            Triangulation<dim> new_cube_9;
            GridGenerator::hyper_rectangle(new_cube_0,
                                           pointA_0,
                                           pointB_0,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_1,
                                           pointA_1,
                                           pointB_1,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_2,
                                           pointA_2,
                                           pointB_2,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_3,
                                           pointA_3,
                                           pointB_3,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_4,
                                           pointA_4,
                                           pointB_4,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_5,
                                           pointA_5,
                                           pointB_5,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_6,
                                           pointA_6,
                                           pointB_6,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_7,
                                           pointA_7,
                                           pointB_7,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_8,
                                           pointA_8,
                                           pointB_8,
                                           false);
            GridGenerator::hyper_rectangle(new_cube_9,
                                           pointA_9,
                                           pointB_9,
                                           false);
            GridGenerator::merge_triangulations({&new_cube_0,
                                                 &new_cube_1,
                                                 &new_cube_2,
                                                 &new_cube_3,
                                                 &new_cube_4,
                                                 &new_cube_5,
                                                 &new_cube_6,
                                                 &new_cube_7,
                                                 &new_cube_8,
                                                 &new_cube_9},
                                                triangulation,
                                                1.0e-10,
                                                false,
                                                false);

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[2] - 0) < 1e-9) // -ve Z
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[2] - total_length) < 1e-9) // +ve Z
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[0] - width) < 1e-9) // +ve X
                        face->set_boundary_id(3);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(4);
                    else if (std::fabs(face_center[1] - width) < 1e-9) // +ve Y
                        face->set_boundary_id(5);
                }

            triangulation.refine_global(2);
            return;
        }

        static void generate_Plate_1(Triangulation<dim> &triangulation)
        {
            double width = 1.0;
            double height = 1.0;

            const Point<dim> p1(0, 0);
            const Point<dim> p2(width, height);

            GridGenerator::hyper_rectangle(triangulation, p1, p2);

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - width) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - height) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                }

            triangulation.refine_global(5);
            return;
        }

        static void generate_Plate_2(Triangulation<dim> &triangulation, std::vector<unsigned int> repetitions, Point<dim> p1, Point<dim> p2)
        {
            GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0] - p1[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - p2[0]) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1] - p1[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - p2[1]) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                }

            return;
        }

        static void generate_Cuboid_1(Triangulation<dim> &triangulation)
        {
            double width_x = 1.0;
            double width_y = 0.1;
            double height_z = 1.0;

            const Point<dim> p1(0, 0, 0);
            const Point<dim> p2(width_x, width_y, height_z);
            std::vector<unsigned int> repetitions = {10, 1, 10};

            GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[2] - 0) < 1e-9) // -ve Z
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[2] - height_z) < 1e-9) // +ve Z
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[0] - width_x) < 1e-9) // +ve X
                        face->set_boundary_id(3);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(4);
                    else if (std::fabs(face_center[1] - width_y) < 1e-9) // +ve Y
                        face->set_boundary_id(5);
                }

            triangulation.refine_global(2);
            return;
        }

        static void generate_Cuboid_2(Triangulation<dim> &triangulation, std::vector<unsigned int> repetitions, Point<dim> p1, Point<dim> p2)
        {
            GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0] - p1[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - p2[0]) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1] - p1[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - p2[1]) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                    else if (std::fabs(face_center[2] - p1[2]) < 1e-9) // -ve Z
                        face->set_boundary_id(4);
                    else if (std::fabs(face_center[2] - p2[2]) < 1e-9) // +ve Z
                        face->set_boundary_id(5);
                }

            // triangulation.refine_global(2);
            return;
        }

        static void read_Plate_mesh_1(Triangulation<dim> &triangulation)
        {
            double width = 1.0;
            double height = 1.0;

            GridIn<dim> grid_in;
            grid_in.attach_triangulation(triangulation);

            std::ifstream input_file("meshes/plate_2D_quad_010.msh");
            grid_in.read_msh(input_file); // Reading mesh from a Gmsh .msh file

            // // Optionally, output the mesh to visually verify it
            // std::ofstream out("read_mesh.vtk");
            // GridOut grid_out;
            // grid_out.write_vtk(triangulation, out);
            // std::cout << "Mesh successfully read and written to 'read_mesh.vtk'" << std::endl;

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - width) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - height) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                }

            // triangulation.refine_global(5);
            return;
        }

        static void read_Plate_mesh_2(Triangulation<dim> &triangulation)
        {
            double width = 1.0;
            double height = 1.0;

            GridIn<dim> grid_in;
            grid_in.attach_triangulation(triangulation);

            std::ifstream input_file("meshes/plate_2D_quad_012.msh");
            grid_in.read_msh(input_file); // Reading mesh from a Gmsh .msh file

            // // Optionally, output the mesh to visually verify it
            // std::ofstream out("read_mesh.vtk");
            // GridOut grid_out;
            // grid_out.write_vtk(triangulation, out);
            // std::cout << "Mesh successfully read and written to 'read_mesh.vtk'" << std::endl;

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - width) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - height) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                }

            // triangulation.refine_global(2);
            return;
        }

        static void read_Cuboid_mesh_1(Triangulation<dim> &triangulation)
        {
            double l_x = 1.0;
            double l_y = 0.1;
            double l_z = 1.0;

            GridIn<dim> grid_in;
            grid_in.attach_triangulation(triangulation);

            std::ifstream input_file("meshes/plate_3D_quad_001.msh");
            grid_in.read_msh(input_file); // Reading mesh from a Gmsh .msh file

            // // Optionally, output the mesh to visually verify it
            // std::ofstream out("read_mesh.vtk");
            // GridOut grid_out;
            // grid_out.write_vtk(triangulation, out);
            // std::cout << "Mesh successfully read and written to 'read_mesh.vtk'" << std::endl;

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - l_x) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - l_y) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                    else if (std::fabs(face_center[2]) < 1e-9) // -ve Z
                        face->set_boundary_id(4);
                    else if (std::fabs(face_center[2] - l_z) < 1e-9) // +ve Z
                        face->set_boundary_id(5);
                }

            // triangulation.refine_global(5);
            return;
        }

        static void read_Cuboid_mesh_2(Triangulation<dim> &triangulation)
        {
            double l_x = 1.0;
            double l_y = 1.0;
            double l_z = 1.0;

            GridIn<dim> grid_in;
            grid_in.attach_triangulation(triangulation);

            std::ifstream input_file("meshes/cuboid_3D_quad_005.msh");
            grid_in.read_msh(input_file); // Reading mesh from a Gmsh .msh file

            // // Optionally, output the mesh to visually verify it
            // std::ofstream out("read_mesh.vtk");
            // GridOut grid_out;
            // grid_out.write_vtk(triangulation, out);
            // std::cout << "Mesh successfully read and written to 'read_mesh.vtk'" << std::endl;

            for (const auto &face : triangulation.active_face_iterators())
                if (face->at_boundary())
                {
                    const Point<dim> face_center = face->center();

                    if (std::fabs(face_center[0]) < 1e-9) // -ve X
                        face->set_boundary_id(0);
                    else if (std::fabs(face_center[0] - l_x) < 1e-9) // +ve X
                        face->set_boundary_id(1);
                    else if (std::fabs(face_center[1]) < 1e-9) // -ve Y
                        face->set_boundary_id(2);
                    else if (std::fabs(face_center[1] - l_y) < 1e-9) // +ve Y
                        face->set_boundary_id(3);
                    else if (std::fabs(face_center[2]) < 1e-9) // -ve Z
                        face->set_boundary_id(4);
                    else if (std::fabs(face_center[2] - l_z) < 1e-9) // +ve Z
                        face->set_boundary_id(5);
                }

            // triangulation.refine_global(5);
            return;
        }
    };

}

#endif // MESH_GENERATOR_H
