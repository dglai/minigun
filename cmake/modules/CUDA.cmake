# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  message(STATUS "Build with CUDA support")

  list(APPEND MINIGUN_LINKER_LIBS ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
  #list(APPEND DGL_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  #list(APPEND DGL_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  #list(APPEND DGL_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
endif(USE_CUDA)
