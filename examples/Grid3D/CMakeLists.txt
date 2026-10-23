# configure_file("settings.json" "settings.json" COPYONLY)
include_directories(PlantModel)
add_executable(grid_check grid_check.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(grid_check PRIVATE vtu11::vtu11)
target_link_libraries(grid_check PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial)
