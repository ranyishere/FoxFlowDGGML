# configure_file("settings.json" "settings.json" COPYONLY)
include_directories(PlantModel)
add_executable(parallel grid_check.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(parallel PRIVATE vtu11::vtu11)
target_link_libraries(parallel PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)
