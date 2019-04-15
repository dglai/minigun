project(benchmark)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(benchmark main.cu)
  target_link_libraries(benchmark ${MINIGUN_LINKER_LIBS} dmlc)
endif()
