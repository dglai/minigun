project(spmv)
message("-- Project added: ${PROJECT_NAME}")

CUDA_ADD_EXECUTABLE(spmv_full full.cu)
target_link_libraries(spmv_full ${MINIGUN_LINKER_LIBS})
CUDA_ADD_EXECUTABLE(spmv_partial partial.cu)
target_link_libraries(spmv_partial ${MINIGUN_LINKER_LIBS})
