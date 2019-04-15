project(benchmark)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(bench_masked_mm bench_masked_mm.cu)
  target_link_libraries(bench_masked_mm ${MINIGUN_LINKER_LIBS} dmlc)
endif()
