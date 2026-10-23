#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <map>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>

#include "DGGML.h"
#include "DggFactory.hpp"
#include "simdjson.h"

#include "ExpandedComplex3D.hpp"
#include "ExpandedComplex2D.hpp"

#include "VtkWriter.hpp"

#include "YAGL_Graph.hpp"
#include "YAGL_Graph_Parallel.hpp"
#include "YAGL_Node.hpp"
#include "SpatialData3D.hpp"

#include "DggFactory.hpp"
#include "SimulationRunner.h"



// This is essentially preferential attachment
// https://en.wikipedia.org/wiki/Preferential_attachment
struct Parameters {
    std::size_t NUM_STEPS = 10;
    double TIME_STEP = 0.08;
    float DELTA = 0.4;
    float MAXIMAL_REACTION_RADIUS = 0.5;

    int MIN_DELTA_STEPS = 1;
    float DELTA_DELTA_T = DELTA / MIN_DELTA_STEPS;
    float DELTA_T_MIN = DELTA_DELTA_T;

    size_t CELL_NX = 1.0;
    size_t CELL_NY = 1.0;
    size_t CELL_NZ = 1.0;
    double CELL_DX = 6.0;
    double CELL_DY = 6.0;
    double CELL_DZ = 6.0;

    // Expanded complex epsilon: must satisfy 3.5*varepsilon < CELL_DX/2
    // Default: varepsilon = CELL_DX / 10.0 satisfies the constraint for any CELL_DX
    double varepsilon = CELL_DX / 30.0;

    // Barabási–Albert growth parameters
    // attach_rate: base multiplier for the preferential-attachment rule.
    //   Effective per-node rate = attach_rate * degree, so the SSA selects an
    //   existing node to receive the new edge with probability degree/Sigma(degree).
    //   NOTE: the network grows exponentially (dN/dt ~ 2*attach_rate*N), so the
    //   final node count is roughly N0 * exp(2*attach_rate*T_total) with
    //   T_total ~ NUM_STEPS * DELTA. Tune attach_rate / NUM_STEPS for the size
    //   you want; a few thousand nodes already give a clear power-law tail.
    double attach_rate = 0.9;
    int    seed_nodes  = 3;     // m0: size of the initial connected seed path
    double step_size   = 0.12;  // seed-node spacing (geometry is cosmetic only)
};

struct Boundary {};
struct Start {};
// degree = number of edges currently incident to this node.  Stored explicitly
// so propensity lambdas can read it from the induced subgraph without access
// to the full system graph.
struct Inner { int degree = 1; };
struct BoundaryInner {};

using key_type = std::size_t;
// using graph_type = YAGL::ParallelGraph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;
using graph_type = YAGL::Graph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;

using CplexGraph3D_t = typename DGGML::ExpandedComplex3D<>::types::graph_type;
using grammar_type = DGGML::Grammar<graph_type>;

// ----------------------------------------------------------------
// Barabási–Albert preferential attachment
//   LHS : Inner(1)                  [any existing network node]
//   RHS : Inner(1) -- Inner(2)      [a brand-new leaf attaches to node 1]
//
// The whole model is this single rule.  Its propensity is LINEAR in the
// matched node's current degree, which is exactly the Barabási–Albert
// attachment kernel
//                       Π(k_i) = k_i / Σ_j k_j .
// Because the SSA selects among all single-node matches in proportion to
// their propensity, an existing node i receives the incoming edge with
// probability k_i / Σ_j k_j — high-degree hubs grow faster, producing a
// scale-free network whose degree distribution follows P(k) ~ k^{-3}.
//
// Each event adds one new node carrying exactly one edge (the m = 1 BA
// model), so after T events the network has N0 + T nodes and N0 + T - 1
// edges.  Degree is tracked in Inner::degree and kept in sync here: the
// new leaf starts at degree 1 and the chosen hub is incremented by 1.
// ----------------------------------------------------------------
void preferential_attachment_rule(grammar_type& grammar, const Parameters& settings)
{
    graph_type lhs, rhs;
    lhs.addNode({1, {Inner{}}});            // any existing node may be chosen

    rhs.addNode({1, {Inner{}}});            // chosen hub (degree incremented)
    rhs.addNode({2, {Inner{}}});            // new leaf node (degree 1)
    rhs.addEdge(1, 2);                      // the single preferential edge

    DGGML::WithRule<graph_type> rule(
        "preferential_attachment", lhs, rhs,
        // Linear preferential-attachment kernel: rate ∝ current degree.
        [settings](auto& lhs, auto& m) -> double {
            int k = lhs[m[1]].template getData<Inner>().degree;
            return settings.attach_rate * static_cast<double>(k);
        },
        [settings](auto& lhs, auto& rhs, auto& m1, auto& m2) {
            // Keep the stored degree consistent with the topology change.
            rhs[m2[2]].template getData<Inner>().degree = 1;                       // new leaf
            rhs[m2[1]].template getData<Inner>().degree =
                lhs[m1[1]].template getData<Inner>().degree + 1;                   // hub + 1

            // Geometry does not affect the BA degree distribution (a single-node
            // LHS has no spatial / reaction-radius constraint), so simply drop the
            // new node at a random interior point.  This keeps every node inside
            // the domain and always matchable, with no positional bias on the
            // attachment dynamics.
            std::mt19937 rng(std::random_device{}());
            const double pad = settings.MAXIMAL_REACTION_RADIUS;
            std::uniform_real_distribution<double> ux(pad, settings.CELL_NX * settings.CELL_DX - pad);
            std::uniform_real_distribution<double> uy(pad, settings.CELL_NY * settings.CELL_DY - pad);
            std::uniform_real_distribution<double> uz(pad, settings.CELL_NZ * settings.CELL_DZ - pad);
            rhs[m2[2]].position[0] = ux(rng);
            rhs[m2[2]].position[1] = uy(rng);
            rhs[m2[2]].position[2] = uz(rng);
        }
    );
    grammar.addRule(rule);
}


class TestModel : public DGGML::Model3D<grammar_type> {

public:

    void initialize() override
    {
        std::cout << "Initializing walking graph simulation\n";
        name = "WalkingGraph3D";

        geoplex2D.init(this->settings.CELL_NX, this->settings.CELL_NY, this->settings.CELL_NZ,
                      this->settings.CELL_DX, this->settings.CELL_DY, this->settings.CELL_DZ,
                       false,
                       settings.varepsilon);

        // Seed: a small connected path of m0 = seed_nodes Inner nodes placed
        // near the domain centre.  Path endpoints start at degree 1 and any
        // interior seed nodes at degree 2; the stored Inner::degree is set from
        // the real graph topology so the preferential-attachment kernel is
        // correct from the very first event.
        const double cx = 0.5 * settings.CELL_NX * settings.CELL_DX;
        const double cy = 0.5 * settings.CELL_NY * settings.CELL_DY;
        const double cz = 0.5 * settings.CELL_NZ * settings.CELL_DZ;

        const int m0 = std::max(2, settings.seed_nodes);
        std::vector<key_type> seed_keys;
        seed_keys.reserve(m0);
        for (int i = 0; i < m0; ++i) {
            const double x = cx + (i - 0.5 * (m0 - 1)) * settings.step_size;
            auto k = gen.get_key();
            system_graph.addNode({k, {Inner{}, x, cy, cz}});
            seed_keys.push_back(k);
        }
        for (int i = 0; i + 1 < m0; ++i)
            system_graph.addEdge(seed_keys[i], seed_keys[i + 1]);

        // Initialise each seed node's stored degree from its real topology.
        for (auto k : seed_keys) {
            auto deg = system_graph.degree(system_graph.findNode(k)->second);
            system_graph[k].template getData<Inner>().degree = static_cast<int>(deg);
        }

        DGGML::GridFileWriter3D grid_writer;
        grid_writer.save({geoplex2D.reaction_grid, geoplex2D.dim_label},
                         "final_reaction_grid");

        preferential_attachment_rule(gamma, settings);
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
        // writer.save(model->geoplex2D.getGraph(), results_dir_name+"/factory_geoplex");
        // DGGML::GridFileWriter grid_writer;
        // grid_writer.save({geoplex2D.reaction_grid,geoplex2D.dim_label},
        // results_dir_name+"/expanded_cell_complex");


        //
        std::string title = results_dir_name+"/simulation_step_";
        std::cout << "Saving the initial state of the system graph\n";
        DGGML::VtkFileWriter<graph_type> vtk_writer;
        vtk_writer.save(system_graph, title+std::to_string(step));
        collect(step);

    }

    // Dump the degree distribution of the current network so the power law can
    // be plotted (see plot_degree_distribution.py).  Degree is read straight
    // from the graph topology, independent of the Inner::degree bookkeeping.
    void collect(std::size_t step) override
    {
        std::map<int, std::size_t> hist;
        std::size_t n = 0;
        for (auto& [key, node] : system_graph.getNodeSetRef()) {
            const int d = static_cast<int>(system_graph.degree(node));
            ++hist[d];
            ++n;
        }

        std::filesystem::create_directories("results");
        const std::string fname =
            "results/degree_distribution_step_" + std::to_string(step) + ".csv";
        std::ofstream out(fname);
        out << "degree,count,pk\n";
        for (const auto& [d, c] : hist)
            out << d << "," << c << ","
                << (n ? static_cast<double>(c) / static_cast<double>(n) : 0.0) << "\n";

        const int kmax = hist.empty() ? 0 : hist.rbegin()->first;
        std::cout << "[collect] step " << step << ": N=" << n
                  << " nodes, max_degree=" << kmax
                  << " -> wrote " << fname << "\n";
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

    // DGGML::SimulationRunnerParallel<TestModel, SIM_DIM> runner(model, true);
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


        // Print out total cell count
        std::cout << "Total reaction grid cells: " << runner.model->geoplex2D.reaction_grid.totalNumCells() << "\n";

        // grid_writer.save({geoplex2D.reaction_grid,geoplex2D.dim_label},
        // results_dir_name+"/expanded_cell_complex");

    // visualize_grid_to_vtk(runner.model->geoplex2D);

    return 0;
}
