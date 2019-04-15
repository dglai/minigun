add_subdirectory(test_graph)
configure_file(data/cora.txt data/cora.txt COPYONLY)
configure_file(data/citeseer.txt data/citeseer.txt COPYONLY)
configure_file(data/pubmed.txt data/pubmed.txt COPYONLY)

project(benchmark)
message("-- Project added: ${PROJECT_NAME}")
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} ${MINIGUN_LINKER_LIBS})
