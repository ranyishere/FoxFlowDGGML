#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <filesystem>
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

#include "SimulationRunner.h"

// ----------------------------------------------------------------------------
// 1D heat equation on a "link" (a chain of vertices):
//
//        du/dt = kappa * d^2u/dx^2,   u(0,t) = u_left,  u(L,t) = u_right
//
// Every vertex carries one temperature u. The PDE is the semi-discrete linear
// finite element (P1, lumped mass) form on the graph, where each edge is an
// element of length h_ij:
//
//        m_i du_i/dt = kappa * sum_{j ~ i} (u_j - u_i) / h_ij ,
//        m_i         = 1/2 * sum_{j ~ i} h_ij
//
// On a uniform chain this reduces to the usual kappa (u_{i-1} - 2u_i + u_{i+1}) / h^2.
// The PDE is one SolvingRule over a single-vertex LHS; it reads the neighbours
// from the system graph, so the discretisation follows the graph topology.
//
// Because the mesh IS the graph, a stochastic rewrite rule can change the mesh
// while the PDE is being solved: "refine_edge" splits an edge by inserting a
// vertex at its midpoint, at a rate proportional to the temperature jump across
// the edge (gradient-driven adaptive mesh refinement), until h reaches h_min.
// ----------------------------------------------------------------------------

struct Parameters {
    std::size_t NUM_STEPS = 20;
    double DELTA = 0.5;
    double DELTA_T_MIN = DELTA / 5.0;
    double MAXIMAL_REACTION_RADIUS = 0.5;

    std::size_t CELL_NX = 1;
    std::size_t CELL_NY = 1;
    std::size_t CELL_NZ = 1;
    double CELL_DX = 6.0;
    double CELL_DY = 6.0;
    double CELL_DZ = 6.0;
    double varepsilon = CELL_DX / 30.0;

    // PDE
    double kappa = 0.1;        // thermal diffusivity
    double u_left = 1.0;       // Dirichlet value at x = 0
    double u_right = 0.0;      // Dirichlet value at x = L
    // the interior starts at u = 0 (needed by the exact solution in collect())

    // Mesh
    double rod_length = 2.0;
    int num_vertices = 21;

    // Adaptive refinement (set refine_rate = 0 for a fixed mesh)
    double refine_rate = 5.0;  // rate = refine_rate * |u_i - u_j| per edge
    double h_min = 0.025;      // edges at or below this length are not split
};

struct Vertex {
    double u = 0.0;
    bool fixed = false;        // Dirichlet vertex: its value is held, not solved
};

using key_type = std::size_t;
using graph_type = YAGL::Graph<key_type, SpatialNode3D<Vertex>>;
using grammar_type = DGGML::Grammar<graph_type>;

template <typename G>
Vertex& vertex(G& g, key_type k) { return g[k].template getData<Vertex>(); }

template <typename G>
double edge_length(G& g, key_type a, key_type b)
{
    double s = 0.0;
    for (int i = 0; i < 3; i++) {
        const double d = g[a].position[i] - g[b].position[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// PDE rule: du_i/dt = kappa/m_i * sum_j (u_j - u_i)/h_ij for every free vertex.
void heat_rule(grammar_type& grammar, const Parameters& settings)
{
    graph_type lhs, rhs;
    lhs.addNode({1, {Vertex{}}});
    rhs.addNode({1, {Vertex{}}});

    const double kappa = settings.kappa;

    DGGML::SolvingRule<graph_type> rule(
        "heat_rule", lhs, rhs, 1,
        // IC: bind the temperature of every free vertex as an ODE variable.
        [](auto& g, auto& m, auto& varset) {
            auto& v = vertex(g, m[1]);
            if (!v.fixed) varset.insert(&v.u);
        },
        // RHS of the semi-discrete PDE.
        [kappa](auto& g, auto& m, auto y, auto ydot, auto& varmap) {
            const key_type key = m[1];
            auto& self = vertex(g, key);
            if (self.fixed) return;

            // Free neighbours are in the state vector; Dirichlet ones are read directly.
            auto read = [&](double* p) {
                auto it = varmap.find(p);
                return it != varmap.end() ? NV_Ith_S(y, it->second) : *p;
            };

            const double ui = read(&self.u);
            double flux = 0.0, mass = 0.0;
            for (auto nk : g.out_neighbors(g.findNode(key)->second)) {
                const double h = edge_length(g, key, nk);
                flux += (read(&vertex(g, nk).u) - ui) / h;
                mass += 0.5 * h;
            }
            if (mass > 0.0)
                NV_Ith_S(ydot, varmap[&self.u]) += kappa * flux / mass;
        });
    grammar.addRule(rule);
}

// Mesh rule: split an edge, inserting a vertex at its midpoint.
//   LHS : (1)---(2)         RHS : (1)---(3)---(2)
void refine_rule(grammar_type& grammar, const Parameters& settings)
{
    graph_type lhs, rhs;
    lhs.addNode({1, {Vertex{}}});
    lhs.addNode({2, {Vertex{}}});
    lhs.addEdge(1, 2);

    rhs.addNode({1, {Vertex{}}});
    rhs.addNode({2, {Vertex{}}});
    rhs.addNode({3, {Vertex{}}});
    rhs.addEdge(1, 3);
    rhs.addEdge(3, 2);

    const double rate = settings.refine_rate;
    const double h_min = settings.h_min;

    DGGML::WithRule<graph_type> rule(
        "refine_edge", lhs, rhs,
        [rate, h_min](auto& lhs, auto& m) -> double {
            if (edge_length(lhs, m[1], m[2]) <= h_min) return 0.0;
            return rate * std::abs(vertex(lhs, m[1]).u - vertex(lhs, m[2]).u);
        },
        [](auto& lhs, auto& rhs, auto& m1, auto& m2) {
            for (int i = 0; i < 3; i++)
                rhs[m2[3]].position[i] = 0.5 * (lhs[m1[1]].position[i] + lhs[m1[2]].position[i]);
            auto& mid = vertex(rhs, m2[3]);
            mid.u = 0.5 * (vertex(lhs, m1[1]).u + vertex(lhs, m1[2]).u);
            mid.fixed = false;
        });
    grammar.addRule(rule);
}

class HeatRodModel : public DGGML::Model3D<grammar_type> {

public:

    void initialize() override
    {
        std::cout << "Initializing 1D heat equation on a link\n";
        name = "HeatRod";

        geoplex2D.init(settings.CELL_NX, settings.CELL_NY, settings.CELL_NZ,
                       settings.CELL_DX, settings.CELL_DY, settings.CELL_DZ,
                       false, settings.varepsilon);

        const double cx = 0.5 * settings.CELL_NX * settings.CELL_DX;
        const double cy = 0.5 * settings.CELL_NY * settings.CELL_DY;
        const double cz = 0.5 * settings.CELL_NZ * settings.CELL_DZ;
        x0 = cx - 0.5 * settings.rod_length;

        const int n = settings.num_vertices;
        const double h = settings.rod_length / (n - 1);
        std::vector<key_type> keys;
        for (int i = 0; i < n; ++i) {
            Vertex v;
            if (i == 0)          { v.fixed = true; v.u = settings.u_left; }
            else if (i == n - 1) { v.fixed = true; v.u = settings.u_right; }
            auto k = gen.get_key();
            system_graph.addNode({k, {v, x0 + i * h, cy, cz}});
            keys.push_back(k);
        }
        for (int i = 0; i + 1 < n; ++i)
            system_graph.addEdge(keys[i], keys[i + 1]);

        heat_rule(gamma, settings);
        if (settings.refine_rate > 0.0)
            refine_rule(gamma, settings);
    }

    void set_parameters(Parameters params) { settings = params; }

    // Series solution for u(x,0)=0 in the interior with the Dirichlet ends held.
    double exact(double x, double t) const
    {
        const double L = settings.rod_length;
        const double xl = x - x0;
        const double uL = settings.u_left, uR = settings.u_right;
        double v = uL + (uR - uL) * xl / L;
        for (int n = 1; n <= 400; ++n) {
            const double c = (n % 2) ? -1.0 : 1.0;
            const double kn = n * M_PI / L;
            const double B = 2.0 / (n * M_PI) * (-uL * (1.0 - c) + (uR - uL) * c);
            v += B * std::sin(kn * xl) * std::exp(-settings.kappa * kn * kn * t);
        }
        return v;
    }

    // Writes x, u, exact for the current mesh and reports the max error.
    void collect(std::size_t step) override
    {
        struct Row { double x, u; };
        std::vector<Row> rows;
        for (auto& [key, node] : system_graph.getNodeSetRef())
            rows.push_back({node.getData().position[0], vertex(system_graph, key).u});
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.x < b.x; });

        const double t = step * settings.DELTA;
        double max_err = 0.0, min_h = 1e300;
        std::filesystem::create_directories("results");
        const std::string fname = "results/heat_step_" + std::to_string(step) + ".csv";
        std::ofstream out(fname);
        out << "x,u,exact\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const double ex = exact(rows[i].x, t);
            out << rows[i].x << "," << rows[i].u << "," << ex << "\n";
            max_err = std::max(max_err, std::abs(rows[i].u - ex));
            if (i > 0) min_h = std::min(min_h, rows[i].x - rows[i - 1].x);
        }
        std::cout << "[collect] step " << step << " t=" << t << ": " << rows.size()
                  << " vertices, min h=" << min_h << ", max |u - exact|=" << max_err
                  << " -> " << fname << "\n";
    }

    void checkpoint(std::size_t step) override
    {
        if (step == 0) {
            std::filesystem::remove_all("results");
            std::filesystem::create_directory("results");
        }
        DGGML::VtkFileWriterComplete<graph_type> vtk_writer;

        // Point data must follow the writer's node order. The writer copies the graph it is
        // given, and a copy keeps the iteration order, so iterate a copy the same way.
        graph_type snapshot = system_graph;
        const double t = step * settings.DELTA;
        std::vector<double> temperature, exact_u;
        for (auto it = snapshot.node_list_begin(); it != snapshot.node_list_end(); ++it) {
            temperature.push_back(vertex(snapshot, it->first).u);
            exact_u.push_back(exact(it->second.getData().position[0], t));
        }
        vtk_writer.set_extra_point_data({{"temperature", temperature}, {"exact", exact_u}});
        vtk_writer.save(snapshot, "results/simulation_step_" + std::to_string(step));
        collect(step);
    }

    Parameters settings;
    double x0 = 0.0;
};

int main(int argc, char* argv[])
{
    const int SIM_DIM = 3;

    HeatRodModel model;
    Parameters params;
    model.set_parameters(params);

    DGGML::SimulationRunner<HeatRodModel, SIM_DIM> runner(model, true);
    runner.initialize();
    runner.run();

    return 0;
}
