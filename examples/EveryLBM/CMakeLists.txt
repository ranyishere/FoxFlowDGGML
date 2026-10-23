add_executable(every_lbm grid_check.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(every_lbm PRIVATE vtu11::vtu11)
target_link_libraries(every_lbm PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)

add_executable(every_lbm_dgg grid_check_dgg.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(every_lbm_dgg PRIVATE vtu11::vtu11)
target_link_libraries(every_lbm_dgg PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)
