add_executable(heat_rod heat_rod.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(heat_rod PRIVATE vtu11::vtu11)
target_link_libraries(heat_rod PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)
