#include <iostream>
#include <chrono>
#include <omp.h>

#include "DGGML.h"
#include "DggFactory.hpp"
#include "simdjson.h"

#include "ExpandedComplex3D.hpp"
#include "ExpandedComplex2D.hpp"

#include "VtkWriter.hpp"

#include "YAGL_Graph.hpp"
#include "YAGL_Node.hpp"
#include "SpatialData3D.hpp"

#include "DggFactory.hpp"
#include "SimulationRunner.h"


struct Parameters {
    std::size_t NUM_STEPS = 100;
    double TIME_STEP = 0.1;
    float DELTA = 0.4;
    float MAXIMAL_REACTION_RADIUS = 0.05;

    int MIN_DELTA_STEPS = 1;
    float DELTA_DELTA_T = DELTA / MIN_DELTA_STEPS;
    float DELTA_T_MIN = DELTA_DELTA_T;

};

struct Boundary {};
struct Start {};
struct Inner {};
struct BoundaryInner {};

using key_type = std::size_t;
using graph_type = YAGL::Graph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;

using CplexGraph3D_t = typename DGGML::ExpandedComplex3D<>::types::graph_type;
using grammar_type = DGGML::Grammar<graph_type>;

template<typename GraphType, typename CplexType, typename ParamType, typename GenType>
void start_node(GraphType &graph, CplexType &cplex, ParamType &settings, GenType &gen) {
    using node_type = typename GraphType::node_type;
    using key_type = typename node_type::key_type;

    node_type start_node = {
        gen.get_key(),
        {
            Start{}, 1.5,1.5,1.5
        }
    };

    graph.addNode(start_node);
};

// Rule 1: A tip GROWS forward — Start → Inner + Start (extends the filament)
void grow_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {Inner{}}});  // old tip becomes inner body
    rhs.addNode({2, {Start{}}});  // new tip pushed forward
    rhs.addEdge(1, 2);

    DGGML::WithRule<graph_type> rule(
        "grow_rule", lhs, rhs,
        [&](auto lhs, auto &m) { return 2.0; }, // fast growth
        [&](auto &lhs, auto &rhs, auto &m1, auto &m2) {
            double step = 0.3;
            rhs[m2[1]].position[0] = lhs[m1[1]].position[0];
            rhs[m2[1]].position[1] = lhs[m1[1]].position[1];
            rhs[m2[1]].position[2] = lhs[m1[1]].position[2];
            // new tip grows in +x direction
            rhs[m2[2]].position[0] = lhs[m1[1]].position[0] + step;
            rhs[m2[2]].position[1] = lhs[m1[1]].position[1];
            rhs[m2[2]].position[2] = lhs[m1[1]].position[2];
        }
    );
    grammar.addRule(rule);
}

// Rule 2: A tip BRANCHES — Start → BoundaryInner + Start + Start (Y-fork)
void branch_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {BoundaryInner{}}});  // junction node
    rhs.addNode({2, {Start{}}});           // branch A
    rhs.addNode({3, {Start{}}});           // branch B
    rhs.addEdge(1, 2);
    rhs.addEdge(1, 3);

    DGGML::WithRule<graph_type> rule(
        "branch_rule", lhs, rhs,
        [&](auto lhs, auto &m) { return 0.3; }, // rare branching
        [&](auto &lhs, auto &rhs, auto &m1, auto &m2) {
            double step = 0.25;
            rhs[m2[1]].position[0] = lhs[m1[1]].position[0];
            rhs[m2[1]].position[1] = lhs[m1[1]].position[1];
            rhs[m2[1]].position[2] = lhs[m1[1]].position[2];
            // Branch A: up-right
            rhs[m2[2]].position[0] = lhs[m1[1]].position[0] + step;
            rhs[m2[2]].position[1] = lhs[m1[1]].position[1] + step;
            rhs[m2[2]].position[2] = lhs[m1[1]].position[2];
            // Branch B: down-right
            rhs[m2[3]].position[0] = lhs[m1[1]].position[0] + step;
            rhs[m2[3]].position[1] = lhs[m1[1]].position[1] - step;
            rhs[m2[3]].position[2] = lhs[m1[1]].position[2];
        }
    );
    grammar.addRule(rule);
}

// Rule 3: A tip RETRACTS — Start → Boundary (tip stalls)
void retract_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {Boundary{}}});  // stalled tip

    DGGML::WithRule<graph_type> rule(
        "retract_rule", lhs, rhs,
        [&](auto lhs, auto &m) { return 0.5; }, // moderate retraction chance
        [&](auto &lhs, auto &rhs, auto &m1, auto &m2) {
            rhs[m2[1]].position[0] = lhs[m1[1]].position[0];
            rhs[m2[1]].position[1] = lhs[m1[1]].position[1];
            rhs[m2[1]].position[2] = lhs[m1[1]].position[2];
        }
    );
    grammar.addRule(rule);
}

// Rule 4: Inner nodes DIFFUSE via ODE (thermal fluctuations along the filament body)
void diffuse_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Inner{}}});

    graph_type rhs;
    rhs.addNode({1, {Inner{}}});

    DGGML::SolvingRule<graph_type> rule(
        "diffuse_rule", lhs, rhs, 3,
        [](auto &lhs, auto &m, auto &varset) {
            varset.insert(&lhs[m[1]].position[0]);
            varset.insert(&lhs[m[1]].position[1]);
            varset.insert(&lhs[m[1]].position[2]);
        },
        [&](auto &lhs, auto &m1, auto y, auto ydot, auto &varmap) {
            double D = 0.01;
            auto x  = NV_Ith_S(y, varmap[&lhs[m1[1]].position[0]]);
            auto yy = NV_Ith_S(y, varmap[&lhs[m1[1]].position[1]]);
            auto z  = NV_Ith_S(y, varmap[&lhs[m1[1]].position[2]]);
            // Damped drift (Ornstein-Uhlenbeck)
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[0]]) += -D * x;
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[1]]) += -D * yy;
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[2]]) += -D * z;
        }
    );
    grammar.addRule(rule);
}


class TestModel : public DGGML::Model3D<grammar_type> {

public:

    void initialize() override
    {
        std::cout << "Initializing filament growth simulation\n";
        name = "FilamentGrowth3D";

        // Larger grid gives filaments more room to grow and branch
        int geoplex_size = 5;
        geoplex2D.init(geoplex_size, geoplex_size, geoplex_size,
                       1.0, 1.0, 1.0,
                       false,  // ghosted
                       0.05);  // maximal reaction radius

        // Seed 5 growing tips spread across the domain so multiple
        // spatial regions are active in parallel from the start
        double positions[5][3] = {
            {0.5, 2.5, 2.5},
            {1.3, 1.0, 1.0},
            {1.3, 4.0, 1.0},
            {1.3, 1.0, 4.0},
            {1.3, 4.0, 4.0}
        };
        for(auto& p : positions)
            system_graph.addNode({gen.get_key(), {Start{}, p[0], p[1], p[2]}});

        grow_rule(gamma, system_graph, settings);
        branch_rule(gamma, system_graph, settings);
        retract_rule(gamma, system_graph, settings);
        diffuse_rule(gamma, system_graph, settings);
    }

    void set_parameters(Parameters params)
    {
        settings = params;
    }

    void checkpoint(std::size_t step) override
    {

        std::cout << "Running the checkpoint for step " << step << "\n";

        std::string results_dir_name = "results";

        // Create the local save directory
        if (step == 0) {
            std::cout << "Cleaning up old results folder if it exists and creating a new one\n";
            std::filesystem::remove_all(results_dir_name);
            std::filesystem::create_directory(results_dir_name);
        }

        // Save expanded cell complex graph
        //DGGML::VtkFileWriter<typename DGGML::ExpandedComplex2D<>::types::graph_type> writer;
        //writer.save(model->geoplex2D.getGraph(), results_dir_name+"/factory_geoplex");
        // ¦   ¦   DGGML::GridFileWriter grid_writer;
        // ¦   ¦   grid_writer.save({geoplex2D.reaction_grid,geoplex2D.dim_label},
        // ¦   ¦   ¦   ¦   ¦   ¦   ¦results_dir_name+"/expanded_cell_complex");
        //
        std::string title = results_dir_name+"/simulation_step_";
        std::cout << "Saving the initial state of the system graph\n";
        DGGML::VtkFileWriter<graph_type> vtk_writer;
        vtk_writer.save(system_graph, title+std::to_string(step));
        collect(step);

    }

    Parameters settings;
    // ExpandedComplex3D<> geoplex3D;
};


void visualize_grid_to_vtk(const DGGML::ExpandedComplex3D<>& geoplex3D)
{

    // For visualizing the 0D, 1D, 2D, 3D cell centers and their connections
    // DGGML::VtkFileWriterComplete<CplexGraph3D_t> graph_writer;
    // graph_writer.write_file("3D_Complex_Graph");

    // For visualizing the underlying fine grid cells (reaction grid) as voxels
    DGGML::GridFileWriter3D grid_writer;
    auto labels = geoplex3D.dim_label;

    std::cout << "Non-zero dimension labels: " << std::endl;
    for (unsigned int i=0; i < labels.size(); i++) {
        if (labels[i] != 0) {
        std::cout << "Dim label[" << i << "] = " << labels[i] << std::endl;
        }
    }

    // exit(0);
    // DGGML::GridFileWriter3D< grid_writer;
    std::pair<DGGML::CartesianGrid3D, std::vector<int>> grid_data =
        std::make_pair(geoplex3D.reaction_grid, geoplex3D.dim_label);

    grid_writer.save(grid_data, "3D_oof");

}

void visualize_grid_to_vtk_2d(const DGGML::ExpandedComplex2D<>& geoplex) {
    DGGML::GridFileWriter grid_writer;
    std::pair<DGGML::CartesianGrid2D, std::vector<int>> grid_data =
        std::make_pair(geoplex.reaction_grid, geoplex.dim_label);

    grid_writer.save(grid_data, "2D_oof");
}

simdjson::ondemand::document load_param_from_json(const std::string& filename) {

    simdjson::ondemand::parser parser;
    simdjson::padded_string json = simdjson::padded_string::load(filename);
    simdjson::ondemand::document settings_file = parser.iterate(json);

    return settings_file;
}

/*
int main(int argc, char* argv[]) {

    std::string param_file = "settings.json";
    // auto model_params = load_param_from_json(param_file);

    const int SIM_DIM = 3;
    DGGML::SimulatorInterface<TestModel, SIM_DIM> simulator;

    TestModel model;

    Parameters model_params;
    model.set_parameters(model_params);

    simulator.setModel(model);

    // simulator.set3D(true);

    simulator.simulate();

}
*/

int main(int argc, char* argv[]) {

    const int SIM_DIM = 3;

    TestModel model;

    Parameters model_params;
    model.set_parameters(model_params);

    // Report how many threads will be used

    DGGML::SimulationRunner<TestModel, SIM_DIM> runner(model, true);

    auto init_start = std::chrono::high_resolution_clock::now();
    runner.initialize();
    auto init_stop = std::chrono::high_resolution_clock::now();
    std::cout << "initialize() took "
              << std::chrono::duration<double, std::milli>(init_stop - init_start).count()
              << " ms\n";

    auto run_start = std::chrono::high_resolution_clock::now();
    runner.run();
    auto run_stop = std::chrono::high_resolution_clock::now();
    std::cout << "run() took "
              << std::chrono::duration<double, std::milli>(run_stop - run_start).count()
              << " ms\n";

    return 0;
}
