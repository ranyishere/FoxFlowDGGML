# configure_file("settings.json" "settings.json" COPYONLY)
include_directories(PlantModel)
add_executable(parallel_particle_swarm_ode grid_check.cpp ../../thirdparty/simdjson/simdjson.cpp)

target_link_libraries(parallel_particle_swarm_ode PRIVATE vtu11::vtu11)
target_link_libraries(parallel_particle_swarm_ode PUBLIC SUNDIALS::arkode SUNDIALS::nvecserial OpenMP::OpenMP_CXX)
