project(traverse)
message("-- Project added: ${PROJECT_NAME}")

CUDA_ADD_EXECUTABLE(succ succ.cu)
target_link_libraries(succ ${MINIGUN_LINKER_LIBS})
