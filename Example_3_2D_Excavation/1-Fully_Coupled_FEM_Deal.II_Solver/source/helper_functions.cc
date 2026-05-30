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

namespace helper_functions_space
{
    using namespace dealii;
    // A function that computes the rotation matrix (2d & 3d implementation)
    // 2d implementation:
    Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>> &grad_u)
    {
        // First, compute the curl of the velocity field from the gradients. Note
        // that we are in 2d, so the rotation is a scalar:
        const double curl = (grad_u[1][0] - grad_u[0][1]);

        // From this, compute the angle of rotation:
        const double angle = std::atan(curl);

        // And from this, build the antisymmetric rotation matrix. We want this
        // rotation matrix to represent the rotation of the local coordinate system
        // with respect to the global Cartesian basis, to we construct it with a
        // negative angle. The rotation matrix therefore represents the rotation
        // required to move from the local to the global coordinate system.
        return Physics::Transformations::Rotations::rotation_matrix_2d(-angle);
    }
    // 3d implementation:
    Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>> &grad_u)
    {
        // Again first compute the curl of the velocity field. This time, it is a
        // real vector:
        const Tensor<1, 3> curl({grad_u[2][1] - grad_u[1][2],
                                 grad_u[0][2] - grad_u[2][0],
                                 grad_u[1][0] - grad_u[0][1]});

        // From this vector, using its magnitude, compute the tangent of the angle
        // of rotation, and from it the actual angle of rotation with respect to
        // the Cartesian basis:
        const double tan_angle = std::sqrt(curl * curl);
        const double angle = std::atan(tan_angle);

        // Now, here's one problem: if the angle of rotation is too small, that
        // means that there is no rotation going on (for example a translational
        // motion). In that case, the rotation matrix is the identity matrix.
        //
        // The reason why we stress that is that in this case we have that
        // <code>tan_angle==0</code>. Further down, we need to divide by that
        // number in the computation of the axis of rotation, and we would get
        // into trouble when dividing doing so. Therefore, let's shortcut this and
        // simply return the identity matrix if the angle of rotation is really
        // small:
        if (std::abs(angle) < 1e-9)
        {
            static const double rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            static const Tensor<2, 3> rot(rotation);
            return rot;
        }

        // Otherwise compute the real rotation matrix. For this, again we rely on
        // a predefined function to compute the rotation matrix of the local
        // coordinate system.
        const Tensor<1, 3> axis = curl / tan_angle;
        return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
                                                                       -angle);
    }

}
