#include <iostream>
#include <chrono>
#include <cmath>
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


struct Parameters {
    std::size_t NUM_STEPS = 100;
    double TIME_STEP = 0.1;
    float DELTA = 0.4;
    float MAXIMAL_REACTION_RADIUS = 0.5;

    int MIN_DELTA_STEPS = 1;
    float DELTA_DELTA_T = DELTA / MIN_DELTA_STEPS;
    float DELTA_T_MIN = DELTA_DELTA_T;

    size_t CELL_NX = 1.0;
    size_t CELL_NY = 1.0;
    size_t CELL_NZ = 1.0;
    double CELL_DX = 3.0;
    double CELL_DY = 3.0;
    double CELL_DZ = 3.0;

    // Expanded complex epsilon: must satisfy 3.5*varepsilon < CELL_DX/2
    // Default: varepsilon = CELL_DX / 10.0 satisfies the constraint for any CELL_DX
    double varepsilon = CELL_DX / 30.0;
};

struct Boundary {};
struct Start {};
struct Inner {};
struct BoundaryInner {};

using key_type = std::size_t;
using graph_type = YAGL::Graph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;
// using graph_type = YAGL::ParallelGraph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;

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

// Swarm rule: particles orbit the centre in a circle that spirals in and out.
// The target orbit radius pulses as R(θ) = R_mean + R_amp·cos(n_petals·θ),
// creating a rose/petal pattern.  A tangential component drives the orbit
// (counter-clockwise in XY) and a radial spring pulls toward the petal radius.
// Swarm rule: particles orbit the centre in XY (rotation around Z axis), helix up/down
// (rotation on axis), and pulse radially as the helix phase changes (in/out pulse).
//
// Three coupled motions:
//   1. Tangential (CCW) component  → rotation around the Z axis
//   2. Helical: z target = Z_amp·sin(n_helix·θ)  → rotates on axis (corkscrew)
//   3. Radial pulse: R_target = R_mean + R_amp·sin(k_phase·(z-cz))
//      Coupling z↔r means as a particle spirals up it expands and as it comes
//      down it contracts, giving continuous in/out pulsation.
void swarm_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {Start{}}});

    const double step     = 0.08;   // max displacement per event
    const double rate     = 10.0;   // event rate

    const double cx = 0.5 * settings.CELL_NX * settings.CELL_DX;
    const double cy = 0.5 * settings.CELL_NY * settings.CELL_DY;
    const double cz = 0.5 * settings.CELL_NZ * settings.CELL_DZ;

    const double R_mean   = 0.35 * 0.9 * std::min(settings.CELL_NX * settings.CELL_DX,
                                                    settings.CELL_NY * settings.CELL_DY);
    const double R_amp    = 0.50 * R_mean;  // radial pulse amplitude
    const double Z_amp    = 0.40 * R_mean;  // helix half-height (on-axis rotation)
    const double k_phase  = 4*M_PI / Z_amp;   // z→radial coupling (one full pulse per helix cycle)
    const int    n_helix  = 2;              // lobes of the helix per orbit
    const double k_radial = 0.5;            // radial spring stiffness
    const double k_z      = 1.1;            // vertical spring stiffness
    const double v_tang   = 1.0;            // tangential speed weight

    DGGML::WithRule<graph_type> rule(
        "swarm_rule", lhs, rhs,
        [rate](auto &lhs, auto &m) { return rate; },
        [step, cx, cy, cz, R_mean, R_amp, Z_amp, k_phase, n_helix,
         k_radial, k_z, v_tang]
        (auto &lhs, auto &rhs, auto &m1, auto &m2) {
            double x = lhs[m1[1]].position[0];
            double y = lhs[m1[1]].position[1];
            double z = lhs[m1[1]].position[2];

            double dx_c = x - cx, dy_c = y - cy;
            double r = std::sqrt(dx_c*dx_c + dy_c*dy_c);

            if (r < 1e-6) {
                rhs[m2[1]].position[0] = x + step;
                rhs[m2[1]].position[1] = y;
                rhs[m2[1]].position[2] = z;
                return;
            }

            double theta = std::atan2(dy_c, dx_c);

            // --- 1. Rotation around Z axis (tangential orbit) ---
            double rx = dx_c / r, ry = dy_c / r;  // radial unit vector
            double tx = -ry,      ty =  rx;        // tangential (CCW)

            // --- 2. Rotate on axis: helical Z target ---
            double z_target = cz + Z_amp * std::sin(n_helix * theta);

            // --- 3. Pulse in/out: R_target coupled to z offset ---
            double z_phase  = (z - cz);            // how far above/below centre plane
            double R_target = R_mean + R_amp * std::sin(k_phase * z_phase);

            // Compose velocity
            double vx = v_tang * tx + k_radial * (R_target - r) * rx;
            double vy = v_tang * ty + k_radial * (R_target - r) * ry;
            double vz = k_z   * (z_target - z);

            double vmag = std::sqrt(vx*vx + vy*vy + vz*vz);
            double s    = (vmag > 1e-6) ? step / vmag : 0.0;

            rhs[m2[1]].position[0] = x + s * vx;
            rhs[m2[1]].position[1] = y + s * vy;
            rhs[m2[1]].position[2] = z + s * vz;
        }
    );
    grammar.addRule(rule);
}


class TestModel : public DGGML::Model3D<grammar_type> {

public:

    void initialize() override
    {
        std::cout << "Initializing circular-spiral swarm simulation\n";
        name = "CircularSpiralSwarm3D";

        geoplex2D.init(this->settings.CELL_NX, this->settings.CELL_NY, this->settings.CELL_NZ,
                      this->settings.CELL_DX, this->settings.CELL_DY, this->settings.CELL_DZ,
                       false,
                       settings.varepsilon);

        // Seed N particles scattered around the mean orbit radius at random angles.
        // A small z-offset gives the swarm a 3-D puff; the swarm rule
        // restores all particles to the centre plane over time.
        const double cx     = 0.5 * settings.CELL_NX * settings.CELL_DX;
        const double cy     = 0.5 * settings.CELL_NY * settings.CELL_DY;
        const double cz     = 0.5 * settings.CELL_NZ * settings.CELL_DZ;
        const double R_mean = 0.35 * 0.5 * std::min(settings.CELL_NX * settings.CELL_DX,
                                                      settings.CELL_NY * settings.CELL_DY);
        const int    N      = 300;

        // Simple deterministic spread: evenly spaced angles with a small radial jitter
        const double Z_amp = 0.40 * R_mean;   // match swarm_rule helix half-height
        for (int i = 0; i < N; ++i) {
            double theta  = 2.0 * M_PI * i / N;
            // seed on the helix curve so particles are already in their attractor
            double r      = R_mean;
            double zoff   = Z_amp * std::sin(2.0 * theta);  // n_helix = 2 lobes
            double x = cx + r * std::cos(theta);
            double y = cy + r * std::sin(theta);
            double z = cz + zoff;
            system_graph.addNode({gen.get_key(), {Start{}, x, y, z}});
        }

        DGGML::GridFileWriter3D grid_writer;
        grid_writer.save({geoplex2D.reaction_grid, geoplex2D.dim_label},
                         "final_reaction_grid");

        swarm_rule(gamma, system_graph, settings);
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
