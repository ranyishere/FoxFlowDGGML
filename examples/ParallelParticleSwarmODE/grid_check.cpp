#include <iostream>
#include <chrono>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>
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

    // 
    size_t CELL_NX = 3.0;
    size_t CELL_NY = 3.0;
    size_t CELL_NZ = 3.0;

    double CELL_DX = 1.0;
    double CELL_DY = 1.0;
    double CELL_DZ = 1.0;

    // Expanded complex epsilon: must satisfy 3.5*varepsilon < CELL_DX/2
    // Default: varepsilon = CELL_DX / 10.0 satisfies the constraint for any CELL_DX
    double varepsilon = CELL_DX / 30.0;

    // VTK output controls (for ParaView). WRITE_VTK is off by default so the
    // statistical A/B harness stays fast; enable it (and set RESULTS_DIR) only for
    // runs you want to inspect. RESULTS_DIR selects the output folder, letting
    // different runs be saved side by side instead of overwriting each other.
    bool WRITE_VTK = false;
    std::string RESULTS_DIR = "results";
};

struct Boundary {};
struct Start {};
struct Inner {};
struct BoundaryInner {};

using key_type = std::size_t;
// using graph_type = YAGL::Graph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;
using graph_type = YAGL::ParallelGraph<key_type, SpatialNode3D<Boundary, Start, Inner, BoundaryInner>>;

using CplexGraph3D_t = typename DGGML::ExpandedComplex3D<>::types::graph_type;
using grammar_type = DGGML::Grammar<graph_type>;

struct PositionStats {
    std::size_t count = 0;
    std::array<double, 3> mean{0.0, 0.0, 0.0};
    std::array<double, 3> stddev{0.0, 0.0, 0.0};
    std::array<double, 3> min{0.0, 0.0, 0.0};
    std::array<double, 3> max{0.0, 0.0, 0.0};
};

struct HistDiff {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
};

using Position = std::array<double, 3>;

struct RunResult {
    std::vector<Position> positions;
    PositionStats stats;
    double init_ms = 0.0;
    double run_ms = 0.0;
    int reaction_cells = 0;
};

struct DistributionMetrics {
    double ks_x = 0.0;
    double ks_y = 0.0;
    double ks_z = 0.0;
    HistDiff hist;
};

struct ScalarSummary {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct MetricsSummary {
    ScalarSummary ks_x;
    ScalarSummary ks_y;
    ScalarSummary ks_z;
    ScalarSummary hist_l1;
    ScalarSummary hist_l2;
    ScalarSummary hist_linf;
};

PositionStats compute_position_stats(const std::vector<Position>& positions) {
    PositionStats out;
    out.count = positions.size();
    if (positions.empty()) return out;

    out.min = positions.front();
    out.max = positions.front();

    for (const auto& p : positions) {
        for (int d = 0; d < 3; ++d) {
            out.mean[d] += p[d];
            out.min[d] = std::min(out.min[d], p[d]);
            out.max[d] = std::max(out.max[d], p[d]);
        }
    }

    for (int d = 0; d < 3; ++d) {
        out.mean[d] /= static_cast<double>(positions.size());
    }

    std::array<double, 3> var{0.0, 0.0, 0.0};
    for (const auto& p : positions) {
        for (int d = 0; d < 3; ++d) {
            double diff = p[d] - out.mean[d];
            var[d] += diff * diff;
        }
    }
    for (int d = 0; d < 3; ++d) {
        var[d] /= static_cast<double>(positions.size());
        out.stddev[d] = std::sqrt(var[d]);
    }

    return out;
}

double ks_distance_1d(std::vector<double> a, std::vector<double> b) {
    if (a.empty() || b.empty()) return 0.0;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());

    std::size_t i = 0, j = 0;
    double dmax = 0.0;
    const double na = static_cast<double>(a.size());
    const double nb = static_cast<double>(b.size());

    while (i < a.size() && j < b.size()) {
        double x = (a[i] < b[j]) ? a[i] : b[j];
        while (i < a.size() && a[i] <= x) ++i;
        while (j < b.size() && b[j] <= x) ++j;
        double fa = static_cast<double>(i) / na;
        double fb = static_cast<double>(j) / nb;
        dmax = std::max(dmax, std::abs(fa - fb));
    }

    return dmax;
}

std::vector<double> normalized_hist_3d(const std::vector<Position>& positions,
                                       int bins_per_dim,
                                       const Parameters& p) {
    const std::size_t total_bins = static_cast<std::size_t>(bins_per_dim) * bins_per_dim * bins_per_dim;
    std::vector<double> hist(total_bins, 0.0);
    if (positions.empty()) return hist;

    const double lx = p.CELL_NX * p.CELL_DX;
    const double ly = p.CELL_NY * p.CELL_DY;
    const double lz = p.CELL_NZ * p.CELL_DZ;

    for (const auto& pt : positions) {
        auto clamp_index = [bins_per_dim](double x, double maxv) {
            if (maxv <= 0.0) return 0;
            int idx = static_cast<int>(std::floor((x / maxv) * bins_per_dim));
            if (idx < 0) idx = 0;
            if (idx >= bins_per_dim) idx = bins_per_dim - 1;
            return idx;
        };

        const int ix = clamp_index(pt[0], lx);
        const int iy = clamp_index(pt[1], ly);
        const int iz = clamp_index(pt[2], lz);

        const std::size_t flat = static_cast<std::size_t>(ix)
                               + static_cast<std::size_t>(bins_per_dim) * static_cast<std::size_t>(iy)
                               + static_cast<std::size_t>(bins_per_dim) * static_cast<std::size_t>(bins_per_dim) * static_cast<std::size_t>(iz);
        hist[flat] += 1.0;
    }

    const double inv_n = 1.0 / static_cast<double>(positions.size());
    for (double& v : hist) v *= inv_n;
    return hist;
}

HistDiff compare_hists(const std::vector<double>& a, const std::vector<double>& b) {
    HistDiff out;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(a[i] - b[i]);
        out.l1 += d;
        out.l2 += d * d;
        out.linf = std::max(out.linf, d);
    }
    out.l2 = std::sqrt(out.l2);
    return out;
}

ScalarSummary summarize_scalars(const std::vector<double>& values) {
    ScalarSummary out;
    if (values.empty()) return out;

    out.min = values.front();
    out.max = values.front();
    for (double v : values) {
        out.mean += v;
        out.min = std::min(out.min, v);
        out.max = std::max(out.max, v);
    }
    out.mean /= static_cast<double>(values.size());

    double variance = 0.0;
    for (double v : values) {
        const double diff = v - out.mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(values.size());
    out.stddev = std::sqrt(variance);
    return out;
}

DistributionMetrics compare_position_distributions(const std::vector<Position>& a,
                                                   const std::vector<Position>& b,
                                                   const Parameters& p,
                                                   int bins_per_dim) {
    DistributionMetrics out;

    std::vector<double> ax, ay, az, bx, by, bz;
    ax.reserve(a.size()); ay.reserve(a.size()); az.reserve(a.size());
    bx.reserve(b.size()); by.reserve(b.size()); bz.reserve(b.size());

    for (const auto& pt : a) {
        ax.push_back(pt[0]);
        ay.push_back(pt[1]);
        az.push_back(pt[2]);
    }
    for (const auto& pt : b) {
        bx.push_back(pt[0]);
        by.push_back(pt[1]);
        bz.push_back(pt[2]);
    }

    out.ks_x = ks_distance_1d(ax, bx);
    out.ks_y = ks_distance_1d(ay, by);
    out.ks_z = ks_distance_1d(az, bz);

    const auto ha = normalized_hist_3d(a, bins_per_dim, p);
    const auto hb = normalized_hist_3d(b, bins_per_dim, p);
    out.hist = compare_hists(ha, hb);
    return out;
}

MetricsSummary summarize_metrics(const std::vector<DistributionMetrics>& metrics) {
    MetricsSummary out;
    std::vector<double> ks_x_vals, ks_y_vals, ks_z_vals, hist_l1_vals, hist_l2_vals, hist_linf_vals;
    ks_x_vals.reserve(metrics.size());
    ks_y_vals.reserve(metrics.size());
    ks_z_vals.reserve(metrics.size());
    hist_l1_vals.reserve(metrics.size());
    hist_l2_vals.reserve(metrics.size());
    hist_linf_vals.reserve(metrics.size());

    for (const auto& metric : metrics) {
        ks_x_vals.push_back(metric.ks_x);
        ks_y_vals.push_back(metric.ks_y);
        ks_z_vals.push_back(metric.ks_z);
        hist_l1_vals.push_back(metric.hist.l1);
        hist_l2_vals.push_back(metric.hist.l2);
        hist_linf_vals.push_back(metric.hist.linf);
    }

    out.ks_x = summarize_scalars(ks_x_vals);
    out.ks_y = summarize_scalars(ks_y_vals);
    out.ks_z = summarize_scalars(ks_z_vals);
    out.hist_l1 = summarize_scalars(hist_l1_vals);
    out.hist_l2 = summarize_scalars(hist_l2_vals);
    out.hist_linf = summarize_scalars(hist_linf_vals);
    return out;
}

void print_scalar_summary(const std::string& label, const ScalarSummary& summary) {
    std::cout << label
              << " mean=" << summary.mean
              << " std=" << summary.stddev
              << " min=" << summary.min
              << " max=" << summary.max << "\n";
}

void print_metrics_summary(const std::string& label, const MetricsSummary& summary) {
    std::cout << "\n" << label << "\n";
    print_scalar_summary("  KS-X:", summary.ks_x);
    print_scalar_summary("  KS-Y:", summary.ks_y);
    print_scalar_summary("  KS-Z:", summary.ks_z);
    print_scalar_summary("  Hist-L1:", summary.hist_l1);
    print_scalar_summary("  Hist-L2:", summary.hist_l2);
    print_scalar_summary("  Hist-Linf:", summary.hist_linf);
}

template <typename RunnerType>
std::vector<Position> extract_positions(RunnerType& runner) {
    std::vector<Position> out;
    out.reserve(runner.model->system_graph.getNodeSetRef().size());
    for (auto& [id, node] : runner.model->system_graph.getNodeSetRef()) {
        const auto& pos = node.getData().position;
        out.push_back({pos[0], pos[1], pos[2]});
    }
    return out;
}

RunResult run_parallel_trial(const Parameters& model_params, unsigned seed, const std::string& results_dir = "");
RunResult run_serial_trial(const Parameters& model_params, unsigned seed, const std::string& results_dir = "");

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

// Swarm rule: stochastic helical orbit — tangential CCW drive + radial pulse + helical Z target.
void swarm_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {Start{}}});

    const double step     = 0.08;
    // const double rate     = 10.0;
    const double rate     = 1.0;

    const double cx = 0.5 * settings.CELL_NX * settings.CELL_DX;
    const double cy = 0.5 * settings.CELL_NY * settings.CELL_DY;
    const double cz = 0.5 * settings.CELL_NZ * settings.CELL_DZ;

    const double R_mean   = 0.35 * 0.9 * std::min(settings.CELL_NX * settings.CELL_DX,
                                                    settings.CELL_NY * settings.CELL_DY);
    const double R_amp    = 0.50 * R_mean;
    const double Z_amp    = 0.40 * R_mean;
    const double k_phase  = 4*M_PI / Z_amp;
    const int    n_helix  = 2;
    const double k_radial = 0.5;
    const double k_z      = 1.1;
    const double v_tang   = 1.0;

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
            double rx = dx_c / r, ry = dy_c / r;
            double tx = -ry,      ty =  rx;

            double z_target = cz + Z_amp * std::sin(n_helix * theta);
            double z_phase  = (z - cz);
            double R_target = R_mean + R_amp * std::sin(k_phase * z_phase);

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

// ODE rule: particles rotate around the Z axis at constant angular velocity omega.
//
// The ODE for CCW rigid-body rotation about Z centred at (cx, cy):
//   dx/dt = -omega * (y - cy)
//   dy/dt = +omega * (x - cx)
//   dz/dt =  0                  (pure XY rotation; z stays fixed)
//
// Because this is a SolvingRule the ODE integrator (SUNDIALS ERK) advances
// all matched particles continuously over the DELTA time window, giving
// smooth, deterministic trajectories rather than discrete stochastic jumps.
void rotate_rule(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {Start{}}});

    const double cx    = 0.5 * settings.CELL_NX * settings.CELL_DX;
    const double cy    = 0.5 * settings.CELL_NY * settings.CELL_DY;
    const double omega = 3.0;   // angular velocity (rad / simulation time unit)

    DGGML::SolvingRule<graph_type> rule(
        "rotate_rule", lhs, rhs, 3,
        // IC: register the three position components as ODE variables.
        [](auto &lhs, auto &m, auto &varset) {
            varset.insert(&lhs[m[1]].position[0]);
            varset.insert(&lhs[m[1]].position[1]);
            varset.insert(&lhs[m[1]].position[2]);
        },
        // ODE RHS: rigid-body rotation around Z.
        [cx, cy, omega](auto &lhs, auto &m1, auto y, auto ydot, auto &varmap) {
            double x  = NV_Ith_S(y, varmap[&lhs[m1[1]].position[0]]);
            double yy = NV_Ith_S(y, varmap[&lhs[m1[1]].position[1]]);
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[0]]) += -omega * (yy - cy);
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[1]]) += +omega * (x  - cx);
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[2]]) += 0.0;
        }
    );
    grammar.addRule(rule);
}

// ODE rule: particles rotate around the Y axis at constant angular velocity omega.
//
// The ODE for CCW rigid-body rotation about Y centred at (cx, cz) in the XZ plane:
//   dx/dt = +omega * (z - cz)
//   dy/dt =  0                  (y stays fixed)
//   dz/dt = -omega * (x - cx)
void rotate_rule_y(grammar_type &grammar, graph_type &system_graph, Parameters &settings) {
    graph_type lhs;
    lhs.addNode({1, {Start{}}});

    graph_type rhs;
    rhs.addNode({1, {Start{}}});

    const double cx    = 0.5 * settings.CELL_NX * settings.CELL_DX;
    const double cy    = 0.5 * settings.CELL_NY * settings.CELL_DY;
    const double cz    = 0.5 * settings.CELL_NZ * settings.CELL_DZ;
    const double omega = 2.0;   // angular velocity (rad / simulation time unit)

    DGGML::SolvingRule<graph_type> rule(
        "rotate_rule_y", lhs, rhs, 3,
        // IC: register the three position components as ODE variables.
        [](auto &lhs, auto &m, auto &varset) {
            varset.insert(&lhs[m[1]].position[0]);
            varset.insert(&lhs[m[1]].position[1]);
            varset.insert(&lhs[m[1]].position[2]);
        },
        // ODE RHS: rigid-body rotation around Y.
        [cx, cy, cz, omega](auto &lhs, auto &m1, auto y, auto ydot, auto &varmap) {
            double x  = NV_Ith_S(y, varmap[&lhs[m1[1]].position[0]]);
            double yy = NV_Ith_S(y, varmap[&lhs[m1[1]].position[1]]);
            double z  = NV_Ith_S(y, varmap[&lhs[m1[1]].position[2]]);
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[0]]) += +omega * (z - cz);
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[1]]) += 0.0;
            NV_Ith_S(ydot, varmap[&lhs[m1[1]].position[2]]) += -omega * (x - cx);
        }
    );
    grammar.addRule(rule);
}


class TestModel : public DGGML::Model3D<grammar_type> {

public:

    void initialize() override
    {
        std::cout << "Initializing Z-rotation ODE swarm simulation\n";
        name = "ZRotationSwarm3D";

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
        rotate_rule(gamma, system_graph, settings);
        rotate_rule_y(gamma, system_graph, settings);
    }

    void set_parameters(Parameters params)
    {
        settings = params;
    }

    void checkpoint(std::size_t step) override
    {
        collect(step);

        // Skip all file output unless this run was explicitly asked to write VTK
        // (keeps the statistical A/B harness fast). RESULTS_DIR lets different runs
        // be saved into separate folders for side-by-side viewing in ParaView.
        if (!settings.WRITE_VTK) return;

        const std::string results_dir_name = settings.RESULTS_DIR;
        std::cout << "Checkpoint step " << step << " -> " << results_dir_name << "\n";

        // Create the local save directory (recursively, so nested paths work)
        if (step == 0) {
            std::cout << "Cleaning up old results folder if it exists and creating a new one: "
                      << results_dir_name << "\n";
            std::filesystem::remove_all(results_dir_name);
            std::filesystem::create_directories(results_dir_name);

            // Save this run's reaction grid into its own folder so the voxel grid
            // can be loaded alongside the particle series in ParaView.
            DGGML::GridFileWriter3D grid_writer;
            grid_writer.save({geoplex2D.reaction_grid, geoplex2D.dim_label},
                             results_dir_name + "/reaction_grid");
        }

        std::string title = results_dir_name+"/simulation_step_";
        DGGML::VtkFileWriter<graph_type> vtk_writer;
        vtk_writer.save(system_graph, title+std::to_string(step));
    }

    Parameters settings;
    // ExpandedComplex3D<> geoplex3D;
};

RunResult run_parallel_trial(const Parameters& model_params, unsigned seed, const std::string& results_dir) {
    const int SIM_DIM = 3;
    TestModel model;
    Parameters params = model_params;
    if (!results_dir.empty()) { params.WRITE_VTK = true; params.RESULTS_DIR = results_dir; }
    model.set_parameters(params);

    // Fixed seed for testing: seed every OpenMP worker thread deterministically
    // (base + thread_id) so re-running reproduces the same stochastic stream.
    DGGML::seed_omp_threads(seed);

    DGGML::SimulationRunnerParallel<TestModel, SIM_DIM> runner(model, true);
    auto init_start = std::chrono::high_resolution_clock::now();
    runner.initialize();
    auto init_stop = std::chrono::high_resolution_clock::now();
    auto run_start = std::chrono::high_resolution_clock::now();
    runner.run();
    auto run_stop = std::chrono::high_resolution_clock::now();

    RunResult out;
    out.positions = extract_positions(runner);
    out.stats = compute_position_stats(out.positions);
    out.init_ms = std::chrono::duration<double, std::milli>(init_stop - init_start).count();
    out.run_ms = std::chrono::duration<double, std::milli>(run_stop - run_start).count();
    out.reaction_cells = runner.model->geoplex2D.reaction_grid.totalNumCells();
    return out;
}

RunResult run_serial_trial(const Parameters& model_params, unsigned seed, const std::string& results_dir) {
    const int SIM_DIM = 3;
    TestModel model;
    Parameters params = model_params;
    if (!results_dir.empty()) { params.WRITE_VTK = true; params.RESULTS_DIR = results_dir; }
    model.set_parameters(params);

    // Fixed seed for testing: the serial engine runs on the main thread, so a
    // single deterministic seed makes the whole run reproducible.
    DGGML::set_global_seed(seed);

    DGGML::SimulationRunner<TestModel, SIM_DIM> runner(model, true);
    auto init_start = std::chrono::high_resolution_clock::now();
    runner.initialize();
    auto init_stop = std::chrono::high_resolution_clock::now();
    auto run_start = std::chrono::high_resolution_clock::now();
    runner.run();
    auto run_stop = std::chrono::high_resolution_clock::now();

    RunResult out;
    out.positions = extract_positions(runner);
    out.stats = compute_position_stats(out.positions);
    out.init_ms = std::chrono::duration<double, std::milli>(init_stop - init_start).count();
    out.run_ms = std::chrono::duration<double, std::milli>(run_stop - run_start).count();
    out.reaction_cells = runner.model->geoplex2D.reaction_grid.totalNumCells();
    return out;
}


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
    Parameters model_params;

    // The serial engine gains nothing from multiple geocells (they only exist to
    // split work across threads) yet still pays their overhead, so benchmarking it
    // on the parallel run's NxNxN geocell layout is unfair. Collapse the serial run
    // to a single geocell while preserving:
    //   (a) the total domain length per axis  (N*DX kept constant), and
    //   (b) the absolute reaction-grid voxel size (varepsilon kept constant),
    // so both engines discretize identical physical space into an identical number
    // of reaction voxels (floor(DX/eps)*N is invariant under N->1, DX->N*DX).
    Parameters serial_params = model_params;
    serial_params.CELL_DX = model_params.CELL_NX * model_params.CELL_DX;
    serial_params.CELL_DY = model_params.CELL_NY * model_params.CELL_DY;
    serial_params.CELL_DZ = model_params.CELL_NZ * model_params.CELL_DZ;
    serial_params.CELL_NX = 1;
    serial_params.CELL_NY = 1;
    serial_params.CELL_NZ = 1;
    serial_params.varepsilon = model_params.varepsilon; // absolute voxel size -> same voxel count

    const int num_trials = 5;
    const int bins = 12;

    // Fixed base seed for reproducible testing. Each trial uses base_seed+trial,
    // shared by the parallel and serial run of that trial. Set to 0 / call
    // DGGML::clear_global_seed() to return to non-deterministic behaviour.
    const unsigned base_seed = 12345u;

    std::vector<RunResult> parallel_trials;
    std::vector<RunResult> serial_trials;
    std::vector<DistributionMetrics> parallel_vs_serial;
    std::vector<DistributionMetrics> serial_vs_serial;
    std::vector<DistributionMetrics> parallel_vs_parallel;
    parallel_trials.reserve(num_trials);
    serial_trials.reserve(num_trials);
    parallel_vs_serial.reserve(num_trials);
    if (num_trials > 1) {
        serial_vs_serial.reserve(num_trials - 1);
        parallel_vs_parallel.reserve(num_trials - 1);
    }

    for (int trial = 0; trial < num_trials; ++trial) {
        std::cout << "\n=== Trial " << (trial + 1) << " / " << num_trials << " ===\n";
        const unsigned trial_seed = base_seed + static_cast<unsigned>(trial);
        parallel_trials.push_back(run_parallel_trial(model_params, trial_seed));
        serial_trials.push_back(run_serial_trial(serial_params, trial_seed));

        parallel_vs_serial.push_back(compare_position_distributions(
            parallel_trials.back().positions,
            serial_trials.back().positions,
            model_params,
            bins));

        if (trial > 0) {
            serial_vs_serial.push_back(compare_position_distributions(
                serial_trials[trial - 1].positions,
                serial_trials[trial].positions,
                model_params,
                bins));
            parallel_vs_parallel.push_back(compare_position_distributions(
                parallel_trials[trial - 1].positions,
                parallel_trials[trial].positions,
                model_params,
                bins));
        }
    }

    std::vector<double> parallel_init_ms, parallel_run_ms, serial_init_ms, serial_run_ms;
    parallel_init_ms.reserve(num_trials);
    parallel_run_ms.reserve(num_trials);
    serial_init_ms.reserve(num_trials);
    serial_run_ms.reserve(num_trials);

    for (int trial = 0; trial < num_trials; ++trial) {
        parallel_init_ms.push_back(parallel_trials[trial].init_ms);
        parallel_run_ms.push_back(parallel_trials[trial].run_ms);
        serial_init_ms.push_back(serial_trials[trial].init_ms);
        serial_run_ms.push_back(serial_trials[trial].run_ms);
    }

    const auto parallel_init_summary = summarize_scalars(parallel_init_ms);
    const auto parallel_run_summary = summarize_scalars(parallel_run_ms);
    const auto serial_init_summary = summarize_scalars(serial_init_ms);
    const auto serial_run_summary = summarize_scalars(serial_run_ms);

    const auto pvss_summary = summarize_metrics(parallel_vs_serial);
    const auto svss_summary = summarize_metrics(serial_vs_serial);
    const auto pvpp_summary = summarize_metrics(parallel_vs_parallel);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Parallel vs Non-Parallel Position Distribution Comparison ===\n";
    std::cout << "Trials: " << num_trials << "\n";
    print_scalar_summary("Parallel init ms:", parallel_init_summary);
    print_scalar_summary("Parallel run ms:", parallel_run_summary);
    print_scalar_summary("Serial   init ms:", serial_init_summary);
    print_scalar_summary("Serial   run ms:", serial_run_summary);

    if (!parallel_trials.empty() && !serial_trials.empty()) {
        const auto& last_parallel = parallel_trials.back();
        const auto& last_serial = serial_trials.back();
        std::cout << "\nLast trial particle counts -> parallel: " << last_parallel.stats.count
                  << ", serial: " << last_serial.stats.count << "\n";
        std::cout << "Last Mean(par): [" << last_parallel.stats.mean[0] << ", " << last_parallel.stats.mean[1] << ", " << last_parallel.stats.mean[2] << "]\n";
        std::cout << "Last Mean(ser): [" << last_serial.stats.mean[0] << ", " << last_serial.stats.mean[1] << ", " << last_serial.stats.mean[2] << "]\n";
        std::cout << "Last Std (par): [" << last_parallel.stats.stddev[0] << ", " << last_parallel.stats.stddev[1] << ", " << last_parallel.stats.stddev[2] << "]\n";
        std::cout << "Last Std (ser): [" << last_serial.stats.stddev[0] << ", " << last_serial.stats.stddev[1] << ", " << last_serial.stats.stddev[2] << "]\n";
        std::cout << "Total reaction grid cells (parallel run): " << last_parallel.reaction_cells << "\n";
        std::cout << "Total reaction grid cells (serial run):   " << last_serial.reaction_cells << "\n";
    }

    print_metrics_summary("Parallel vs Serial (same trial index)", pvss_summary);
    if (!serial_vs_serial.empty()) {
        print_metrics_summary("Serial vs Serial baseline (adjacent trials)", svss_summary);
    }
    if (!parallel_vs_parallel.empty()) {
        print_metrics_summary("Parallel vs Parallel baseline (adjacent trials)", pvpp_summary);
    }

    // --- VTK visualization runs (for ParaView) -----------------------------
    // Re-run one parallel and one serial simulation at the same seed, each writing
    // its full simulation_step_*.vtk time series into its own folder so the two can
    // be loaded side by side in ParaView. These runs are independent of the stats
    // above (those ran with VTK output disabled for speed).
    {
        const unsigned vis_seed = base_seed;
        const std::string parallel_dir = "results/parallel_seed_" + std::to_string(vis_seed);
        const std::string serial_dir   = "results/serial_seed_"   + std::to_string(vis_seed);
        std::cout << "\n=== Writing VTK visualization runs (seed " << vis_seed << ") ===\n";
        run_parallel_trial(model_params,  vis_seed, parallel_dir);
        run_serial_trial(serial_params, vis_seed, serial_dir);
        std::cout << "VTK time series written to:\n"
                  << "  " << parallel_dir << "  (parallel, " << model_params.CELL_NX << "^3 geocells)\n"
                  << "  " << serial_dir   << "  (serial, single geocell)\n"
                  << "Open each folder's simulation_step_..vtk series in ParaView.\n";
    }

    // --- Determinism check -------------------------------------------------
    // Run the SAME seed twice and compare. Expectation:
    //   * self vs self     -> KS exactly 0 (sanity: metric returns 0 for identical input).
    //   * Serial   same-seed -> KS at the 1/N floor (serial is deterministic up to
    //     at most a single particle's sub-bin position).
    //   * Parallel same-seed -> KS > 0 (OpenMP scheduling + FP reduction order
    //     are nondeterministic even with identical seeds).
    //   * Parallel vs Serial same-seed -> cross-method divergence at one seed.
    // Only a few repeats are needed since the result is essentially binary;
    // serial runs are expensive, so keep this count small.
    const int determinism_trials = 3;
    std::vector<DistributionMetrics> self_vs_self;
    std::vector<DistributionMetrics> serial_same_seed;
    std::vector<DistributionMetrics> parallel_same_seed;
    std::vector<DistributionMetrics> parallel_vs_serial_same;
    self_vs_self.reserve(determinism_trials);
    serial_same_seed.reserve(determinism_trials);
    parallel_same_seed.reserve(determinism_trials);
    parallel_vs_serial_same.reserve(determinism_trials);

    for (int i = 0; i < determinism_trials; ++i) {
        std::cout << "\n=== Determinism check " << (i + 1) << " / " << determinism_trials << " ===\n";
        const unsigned s = base_seed + 1000u + static_cast<unsigned>(i);

        const RunResult serial_a = run_serial_trial(serial_params, s);
        const RunResult serial_b = run_serial_trial(serial_params, s);

        // Sanity: identical input must yield exactly 0 on every metric.
        self_vs_self.push_back(compare_position_distributions(
            serial_a.positions, serial_a.positions, model_params, bins));

        serial_same_seed.push_back(compare_position_distributions(
            serial_a.positions, serial_b.positions, model_params, bins));

        const RunResult parallel_a = run_parallel_trial(model_params, s);
        const RunResult parallel_b = run_parallel_trial(model_params, s);
        parallel_same_seed.push_back(compare_position_distributions(
            parallel_a.positions, parallel_b.positions, model_params, bins));

        // Cross-method at the SAME seed.
        parallel_vs_serial_same.push_back(compare_position_distributions(
            parallel_a.positions, serial_a.positions, model_params, bins));
    }

    const auto self_summary = summarize_metrics(self_vs_self);
    const auto svss_same_summary = summarize_metrics(serial_same_seed);
    const auto pvpp_same_summary = summarize_metrics(parallel_same_seed);
    const auto pvss_same_summary = summarize_metrics(parallel_vs_serial_same);
    print_metrics_summary("Self vs Self (identical input) -- expect exactly 0", self_summary);
    print_metrics_summary("Serial vs Serial (SAME seed) -- expect ~1/N floor", svss_same_summary);
    print_metrics_summary("Parallel vs Parallel (SAME seed) -- expect KS>0", pvpp_same_summary);
    print_metrics_summary("Parallel vs Serial (SAME seed)", pvss_same_summary);

    std::cout << "\nNOTE: stochastic draws are seeded deterministically (DGGML::set_global_seed/seed_omp_threads).\n"
                 "Self-vs-self is the metric's true zero. Serial-vs-serial sits at the 1/N KS floor (<= one\n"
                 "particle), confirming serial is effectively deterministic; same-seed parallel repeats stay >0\n"
                 "due to scheduling/FP-reduction order. The cross-method signal is whether parallel-vs-serial\n"
                 "metrics consistently exceed the parallel-vs-parallel baseline.\n";

    return 0;
}
