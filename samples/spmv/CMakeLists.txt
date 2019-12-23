project(spmv)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(spmv_full_cuda full.cu)
  target_link_libraries(spmv_full_cuda ${MINIGUN_LINKER_LIBS})
endif()

ADD_EXECUTABLE(spmv_full_cpu full.cc)
target_compile_options(spmv_full_cpu PUBLIC -fopenmp)
target_link_libraries(spmv_full_cpu ${MINIGUN_LINKER_LIBS})
