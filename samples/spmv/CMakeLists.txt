project(spmv)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(spmv_full_cuda full.cu)
  target_link_libraries(spmv_full_cuda ${MINIGUN_LINKER_LIBS})
  CUDA_ADD_EXECUTABLE(spmv_partial_cuda partial.cu)
  target_link_libraries(spmv_partial_cuda ${MINIGUN_LINKER_LIBS})
endif()

ADD_EXECUTABLE(${PROJECT_NAME}_cpu main.cc)
target_compile_options(${PROJECT_NAME}_cpu PUBLIC -fopenmp)
target_link_libraries(${PROJECT_NAME}_cpu ${MINIGUN_LINKER_LIBS})
