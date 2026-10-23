# configure_file("settings.json" "settings.json" COPYONLY)
include_directories(PlantModel)
add_executable(parallel_walking_graph grid_check.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(parallel_walking_graph PRIVATE vtu11::vtu11)
target_link_libraries(parallel_walking_graph PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)
