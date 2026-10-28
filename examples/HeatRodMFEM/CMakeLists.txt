add_executable(heat_rod_mfem heat_rod_mfem.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(heat_rod_mfem PRIVATE vtu11::vtu11 mfem)
target_link_libraries(heat_rod_mfem PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)
