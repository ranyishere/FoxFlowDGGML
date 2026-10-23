#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
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

// ----------------------------------------------------------------------------
// Reaction-diffusion model that converges to the mean-field law of mass action
//   A + B --k--> C ,  with  d[A]/dt = d[B]/dt = -k[A][B],  d[C]/dt = +k[A][B].
//
// Species encoding:  A = Start,  B = BoundaryInner,  C = Inner (inert product).
//
// Mechanisms (each a graph-rewrite rule fired by the stochastic engine):
//   * add_random_walk_rule : every A and B takes an unbiased Brownian step
//     (Gaussian displacement per axis) with reflective domain walls  -> diffusion.
//   * add_bounce_rule      : overlapping same-species pairs are pushed apart
//     (hard-sphere excluded volume)                                  -> bouncing.
//   * combine_rule         : an A and a B within the reaction radius fuse into a
//     single C with a contact-ramped propensity                      -> the A+B->C
//     reaction channel.
// In the dilute, well-mixed limit these spatial dynamics reproduce the ODEs above.
// ----------------------------------------------------------------------------

// Reflect a coordinate back into [0, L] (perfectly elastic walls) so the random
// walk conserves particles instead of letting them escape the domain.
inline double reflect_into(double v, double L) {
    if (L <= 0.0) return v;
    while (v < 0.0 || v > L) {
        if (v < 0.0) v = -v;
        if (v > L)   v = 2.0 * L - v;
    }
    return v;
}

// Unbiased Brownian step for a single species with reflective boundaries.
//   rate  : how often (propensity) a particle attempts a jump
//   sigma : std-dev of the Gaussian displacement per axis (sets the step size)
template <typename SpeciesT>
void add_random_walk_rule(const std::string& name, grammar_type& grammar,
                          Parameters& settings, double rate, double sigma) {
    graph_type lhs; lhs.addNode({1, {SpeciesT{}}});
    graph_type rhs; rhs.addNode({1, {SpeciesT{}}});

    const double Lx = settings.CELL_NX * settings.CELL_DX;
    const double Ly = settings.CELL_NY * settings.CELL_DY;
    const double Lz = settings.CELL_NZ * settings.CELL_DZ;

    DGGML::WithRule<graph_type> rule(
        name, lhs, rhs,
        [rate](auto& lhs, auto& m) { return rate; },
        [sigma, Lx, Ly, Lz](auto& lhs, auto& rhs, auto& m1, auto& m2) {
            std::normal_distribution<double> nd(0.0, sigma);
            auto& rng = DGGML::thread_rng();
            double x = lhs[m1[1]].position[0] + nd(rng);
            double y = lhs[m1[1]].position[1] + nd(rng);
            double z = lhs[m1[1]].position[2] + nd(rng);
            rhs[m2[1]].position[0] = reflect_into(x, Lx);
            rhs[m2[1]].position[1] = reflect_into(y, Ly);
            rhs[m2[1]].position[2] = reflect_into(z, Lz);
        }
    );
    grammar.addRule(rule);
}

// Hard-sphere excluded volume: two same-species particles that overlap within the
// core diameter `sigma` are pushed apart along their separation axis -> they bounce.
// The 2-node pattern is matched on proximity (pairs within MAXIMAL_REACTION_RADIUS),
// so keep sigma < MAXIMAL_REACTION_RADIUS for the pair to be discoverable.
template <typename SpeciesT>
void add_bounce_rule(const std::string& name, grammar_type& grammar,
                     Parameters& settings, double sigma, double rate) {
    graph_type lhs; lhs.addNode({1, {SpeciesT{}}}); lhs.addNode({2, {SpeciesT{}}});
    graph_type rhs; rhs.addNode({1, {SpeciesT{}}}); rhs.addNode({2, {SpeciesT{}}});

    DGGML::WithRule<graph_type> rule(
        name, lhs, rhs,
        [sigma, rate](auto& lhs, auto& m) {
            auto& p1 = lhs[m[1]].position;
            auto& p2 = lhs[m[2]].position;
            double dx = p1[0]-p2[0], dy = p1[1]-p2[1], dz = p1[2]-p2[2];
            double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            // Only fire while the cores overlap (d < sigma); stronger the deeper.
            return rate * std::max(0.0, 1.0 - d / sigma);
        },
        [sigma](auto& lhs, auto& rhs, auto& m1, auto& m2) {
            auto& p1 = lhs[m1[1]].position;
            auto& p2 = lhs[m1[2]].position;
            double dx = p1[0]-p2[0], dy = p1[1]-p2[1], dz = p1[2]-p2[2];
            double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < 1e-9) { dx = 1.0; dy = 0.0; dz = 0.0; d = 1.0; } // coincident -> pick an axis
            const double ux = dx/d, uy = dy/d, uz = dz/d;
            const double push = 0.5 * (sigma - d); // split the separation evenly
            rhs[m2[1]].position[0] = p1[0] + ux*push;
            rhs[m2[1]].position[1] = p1[1] + uy*push;
            rhs[m2[1]].position[2] = p1[2] + uz*push;
            rhs[m2[2]].position[0] = p2[0] - ux*push;
            rhs[m2[2]].position[1] = p2[1] - uy*push;
            rhs[m2[2]].position[2] = p2[2] - uz*push;
        }
    );
    grammar.addRule(rule);
}

// Combine: SpeciesA + SpeciesB -> C (Inner) at their midpoint.
// Propensity ramps linearly with proximity -- zero at reaction_radius, max at contact.
void combine_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});
    lhs.addNode({2, {BoundaryInner{}}});

    graph_type rhs;
    rhs.addNode({1, {Inner{}}});
    // node 2 absent → deleted

    const double reaction_radius = settings.MAXIMAL_REACTION_RADIUS;
    const double k_rate = 20.0;

    DGGML::WithRule<graph_type> rule(
        "combine_rule", lhs, rhs,
        [reaction_radius, k_rate](auto &lhs, auto &m) {
            // Distance between the two matched nodes in the induced subgraph
            auto& p1 = lhs[m[1]].position;
            auto& p2 = lhs[m[2]].position;
            double dx = p1[0]-p2[0], dy = p1[1]-p2[1], dz = p1[2]-p2[2];
            double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            // Linear ramp: full rate at d=0, zero at d=reaction_radius
            return k_rate * std::max(0.0, 1.0 - d / reaction_radius);
        },
        [](auto &lhs, auto &rhs, auto &m1, auto &m2) {
            rhs[m2[1]].position[0] = 0.5*(lhs[m1[1]].position[0] + lhs[m1[2]].position[0]);
            rhs[m2[1]].position[1] = 0.5*(lhs[m1[1]].position[1] + lhs[m1[2]].position[1]);
            rhs[m2[1]].position[2] = 0.5*(lhs[m1[1]].position[2] + lhs[m1[2]].position[2]);
        }
    );
    grammar.addRule(rule);
}


class TestModel : public DGGML::Model3D<grammar_type> {

public:

    void initialize() override
    {
        std::cout << "Initializing two-species diffusion-combination simulation\n";
        name = "TwoSpeciesCombination3D";

        int geoplex_size = 5;
        geoplex2D.init(this->settings.CELL_NX, this->settings.CELL_NY, this->settings.CELL_NZ,
                      this->settings.CELL_DX, this->settings.CELL_DY, this->settings.CELL_DZ,
                       false,
                       settings.varepsilon);

        // Domain: [0, NX*DX] x [0, NY*DY] x [0, NZ*DZ]
        // Keep particles at least 1 cell width (CELL_DX) from every wall.
        // Place 100 SpeciesA on the left face (x = CELL_DX) and
        // 100 SpeciesB on the right face (x = (NX-1)*CELL_DX).
        // Arrange each species as a 10x10 grid on its face in y/z.
        // Domain: [0, NX*DX] x [0, NY*DY] x [0, NZ*DZ].
        // Mean-field (well-mixed) initial condition: scatter N_A copies of A and
        // N_B copies of B uniformly at random throughout the interior. Starting
        // well-mixed (rather than segregated on two faces) is what lets the
        // aggregate kinetics track the mass-action ODEs from t=0.
        const double margin = settings.varepsilon; // keep strictly inside the reaction grid
        const double Lx = settings.CELL_NX * settings.CELL_DX;
        const double Ly = settings.CELL_NY * settings.CELL_DY;
        const double Lz = settings.CELL_NZ * settings.CELL_DZ;

        auto rx = RandomRealsBetween(margin, Lx - margin);
        auto ry = RandomRealsBetween(margin, Ly - margin);
        auto rz = RandomRealsBetween(margin, Lz - margin);

        const int n_A = 250;
        const int n_B = 250;
        for (int i = 0; i < n_A; ++i)
            system_graph.addNode({gen.get_key(), {Start{},         rx(), ry(), rz()}});
        for (int i = 0; i < n_B; ++i)
            system_graph.addNode({gen.get_key(), {BoundaryInner{}, rx(), ry(), rz()}});

        DGGML::GridFileWriter3D grid_writer;
        grid_writer.save({geoplex2D.reaction_grid,geoplex2D.dim_label},
        "final_reaction_grid");

        // --- Rule rates / length scales ------------------------------------
        const double walk_rate  = 8.0;   // diffusion attempt frequency
        const double walk_sigma = 0.10;  // Gaussian step std-dev per axis
        const double core_sigma = 0.15;  // hard-sphere diameter (< MAXIMAL_REACTION_RADIUS)
        const double bounce_rate = 60.0; // stiffness of the excluded-volume push

        // A and B diffuse (random walk with reflective walls)
        add_random_walk_rule<Start>("walk_a", gamma, settings, walk_rate, walk_sigma);
        add_random_walk_rule<BoundaryInner>("walk_b", gamma, settings, walk_rate, walk_sigma);
        // Same-species excluded volume: A bounces off A, B bounces off B.
        // (A and B do not bounce -- on contact they react via combine_rule instead.)
        add_bounce_rule<Start>("bounce_a", gamma, settings, core_sigma, bounce_rate);
        add_bounce_rule<BoundaryInner>("bounce_b", gamma, settings, core_sigma, bounce_rate);
        // A + B -> C when within the reaction radius.
        combine_rule(gamma, system_graph, settings);
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
            std::filesystem::create_directories(results_dir_name);
        }

        // Count species for the mean-field comparison. The variant index encodes
        // the species: Boundary=0, Start(A)=1, Inner(C)=2, BoundaryInner(B)=3.
        std::size_t nA = 0, nB = 0, nC = 0;
        for (auto& [key, node] : system_graph.getNodeSetRef()) {
            switch (node.getData().data.index()) {
                case 1: ++nA; break; // A = Start
                case 3: ++nB; break; // B = BoundaryInner
                case 2: ++nC; break; // C = Inner
                default: break;
            }
        }
        // Append (step, A, B, C) so the trajectory can be plotted against the ODE
        //   d[A]/dt = d[B]/dt = -k[A][B],  d[C]/dt = +k[A][B].
        std::ofstream csv(results_dir_name + "/species_counts.csv",
                          step == 0 ? std::ios::trunc : std::ios::app);
        if (step == 0) csv << "step,A,B,C\n";
        csv << step << ',' << nA << ',' << nB << ',' << nC << '\n';
        std::cout << "  species counts: A=" << nA << "  B=" << nB << "  C=" << nC << "\n";

        std::string title = results_dir_name+"/simulation_step_";
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