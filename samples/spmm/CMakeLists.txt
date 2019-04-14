project(spmm)
message("-- Project added: ${PROJECT_NAME}")

CUDA_ADD_EXECUTABLE(spmm_full full.cu)
target_link_libraries(spmm_full ${MINIGUN_LINKER_LIBS})
