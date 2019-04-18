project(test_graph)
message("-- Project added: ${PROJECT_NAME}")

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} ${MINIGUN_LINKER_LIBS})
