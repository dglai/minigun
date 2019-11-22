cmake_minimum_required(VERSION 2.8)
project(minigun C CXX)

message(STATUS "Start configuring project ${PROJECT_NAME}")

# find packages
include(cmake/util/Util.cmake)
include(cmake/util/MshadowUtil.cmake)
include(cmake/util/FindCUDA.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

minigun_option(USE_CUDA "Build with CUDA" ON)
minigun_option(BUILD_SAMPLES "Build minigun samples" OFF)
minigun_option(ENABLE_PARTIAL_FRONTIER "Enable build with partial frontier support" ON)

# include directories
include_directories("minigun")
include_directories("third_party/dlpack/include")
include_directories("third_party/dmlc-core/include")
include_directories("third_party/moderngpu/src")

set(MINIGUN_LINKER_LIBS "")

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")

  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(MSVC)

include(cmake/modules/CUDA.cmake)

if(ENABLE_PARTIAL_FRONTIER)
  add_definitions(-DENABLE_PARTIAL_FRONTIER)
endif(ENABLE_PARTIAL_FRONTIER)

file(GLOB_RECURSE MINIGUN_SRC "minigun/minigun/*.h")

if(BUILD_SAMPLES)
  add_subdirectory(third_party/dmlc-core)
  add_subdirectory(samples)
endif(BUILD_SAMPLES)
