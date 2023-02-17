cmake_minimum_required(VERSION 3.23)
project(omp101 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(IntelDPCPP_DIR "/opt/intel/oneapi/compiler/2023.0.0/linux/IntelDPCPP")
find_package(IntelDPCPP REQUIRED)

include_directories(${SYCL_INCLUDE_DIR})
link_directories(${SYCL_LIBRARY_DIR})
add_compile_options(-O3 -fiopenmp -fopenmp-targets=spir64)
add_link_options(-fiopenmp -fopenmp-targets=spir64)
link_libraries(iomp5 omptarget)

add_executable(0_matmul ${PROJECT_SOURCE_DIR}/0_matmul.cc)
add_executable(1_parallel_pi ${PROJECT_SOURCE_DIR}/1_parallel_pi.cc)

# add_compile_options(-Werror -Wall -Wextra -pedantic
#     -Wno-unused-command-line-argument
#     -Wno-c11-extensions
#     -O0
#     -ggdb3
#     )

# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag(-Weverything CXX_HAS_EVERYTHING)
# if (CXX_HAS_EVERYTHING)
#     add_compile_options(-Wall -Wextra -Weverything -Werror
#         -Wno-c++98-compat
#         -Wno-c++98-compat-pedantic
#         -Wno-missing-prototypes
#         -Wno-unused-function
#         -Wno-unused-member-function
#         -Wno-unused-variable
#         -Wno-used-but-marked-unused
#         -Wno-padded
#         -Wno-missing-variable-declarations
#         -Wno-global-constructors
#         -Wno-unused-parameter
#         -Wno-error=exit-time-destructors
#         -Wno-error=literal-conversion
#         -Wno-error=padded
#         -Wno-error=suggest-destructor-override
#         -Wno-error=zero-as-null-pointer-constant
#         -Wno-error=weak-vtables
#         -Wno-error=old-style-cast
#         -Wno-error=reserved-macro-identifier
#         -Wno-error=undef
#         -fcolor-diagnostics
#         )
# endif(CXX_HAS_EVERYTHING)
