project(spmm)
message("-- Project added: ${PROJECT_NAME}")

if(USE_CUDA)
  CUDA_ADD_EXECUTABLE(${PROJECT_NAME}_gpu main.cu)
  target_link_libraries(${PROJECT_NAME}_gpu ${MINIGUN_LINKER_LIBS})
endif()

ADD_EXECUTABLE(${PROJECT_NAME}_cpu main.cc)
target_compile_options(${PROJECT_NAME}_cpu PUBLIC -fopenmp)
target_link_libraries(${PROJECT_NAME}_cpu ${MINIGUN_LINKER_LIBS})
