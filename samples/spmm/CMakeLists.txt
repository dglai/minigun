project(spmm)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(spmm_full_cuda full.cu)
  target_link_libraries(spmm_full_cuda ${MINIGUN_LINKER_LIBS})
endif()

ADD_EXECUTABLE(spmm_full_cpu full.cc)
target_compile_options(spmm_full_cpu PUBLIC -fopenmp)
target_link_libraries(spmm_full_cpu ${MINIGUN_LINKER_LIBS})
