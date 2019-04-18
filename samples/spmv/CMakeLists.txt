project(spmv)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(spmv_full_cuda full.cu)
  target_link_libraries(spmv_full_cuda ${MINIGUN_LINKER_LIBS})
  CUDA_ADD_EXECUTABLE(spmv_partial_cuda partial.cu)
  target_link_libraries(spmv_partial_cuda ${MINIGUN_LINKER_LIBS})
endif()

ADD_EXECUTABLE(spmv_full_cpu full.cc)
target_compile_options(spmv_full_cpu PUBLIC -fopenmp)
target_link_libraries(spmv_full_cpu ${MINIGUN_LINKER_LIBS})

ADD_EXECUTABLE(spmv_partial_cpu partial.cc)
target_compile_options(spmv_partial_cpu PUBLIC -fopenmp)
target_link_libraries(spmv_partial_cpu ${MINIGUN_LINKER_LIBS})
