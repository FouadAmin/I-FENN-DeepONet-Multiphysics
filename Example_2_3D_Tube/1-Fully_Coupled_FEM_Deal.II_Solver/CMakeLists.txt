# Set the name of the project and target:
set(TARGET "program-010")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  file(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  file(GLOB_RECURSE TARGET_INC  "include/*.h")
#  set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
# set(TARGET_SRC
#   ${TARGET}.cc
# )

file(GLOB_RECURSE TARGET_SRC  "source/*.cc")
file(GLOB_RECURSE TARGET_INC  "include/*.h")
set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})

# Usually, you will not need to modify anything beyond this point...
  
cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.6.0
HINTS
${deal.II_DIR} ${DEAL_II_DIR} ../ ../../  $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
message(FATAL_ERROR "\n"
"*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
"You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
"or set an environment variable \"DEAL_II_DIR\" that contains this path."
)
endif ()

#
# Are all dependencies fulfilled?
#
if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_PETSC OR DEAL_II_PETSC_WITH_COMPLEX) # keep in one line
message(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_MPI = ${DEAL_II_WITH_MPI}
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
This conflicts with the requirements."
    )
endif()

# Include cnpy header files
include_directories(/usr/local/include)

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()

# Find the cnpy library
find_library(CNPY_LIB cnpy HINTS /usr/local/lib)

# Link the cnpy library
if(CNPY_LIB)
    target_link_libraries(${TARGET} ${CNPY_LIB})
else()
    message(FATAL_ERROR "cnpy library not found.")
endif()
