#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <map>
#include <mutex>
#include <memory>
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

#include "mfem.hpp"

// ----------------------------------------------------------------------------
// MFEM-backed PDE rule: 1D heat equation on a "link" (a chain graph).
//
//        du/dt = kappa * d^2u/dx^2,   u(0,t) = u_left,  u(L,t) = u_right
//
// The spatial discretisation is done by MFEM (P1 H1 space on a 1D mesh,
// M du/dt = -K u with the Dirichlet ends eliminated). The MFEM field lives in
// the ODE state vector of the geocell that hosts the chain:
//
//        y = [ grammar variables | MFEM dofs | tau ]
//
// and is advanced by the same SUNDIALS ERK integrator as every other grammar
// variable, so it interleaves with stochastic rewrites. The grammar side is the
// "heat_pde" SolvingRule: every chain vertex holds the temperature u of its dof
// as a grammar variable and follows du_i/dt = d(dof_i)/dt, so rules (stochastic
// or otherwise) can read the PDE solution straight from the graph.
//
// How the pieces are wired:
//   * the model exposes `fem_operator_type` and `fem_operator(k, has_rules)`;
//     approximate_ssa hands the returned operator to the per-geocell Solver
//   * the Solver sets fem_offset, calls Load() on construction, Mult() in every
//     RHS evaluation and Sync() in copy_back(), so the field persists across the
//     per-window Solver objects
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

    // Smooth initial profile: the steady linear profile plus two sine modes,
    //   u(x,0) = u_left + (u_right - u_left) x/L + sum_n a_n sin(n pi x/L)
    // which satisfies the Dirichlet ends and has an exact solution (see exact())
    std::vector<std::pair<int, double>> modes = {{1, 0.5}, {3, 0.2}};

    // Mesh: num_elements segments => num_elements + 1 vertices / dofs
    double rod_length = 2.0;
    int num_elements = 80;
};

struct Vertex {
    double u = 0.0;   // temperature, mirrors the MFEM dof
    int dof = 0;      // index of this vertex's MFEM dof
};

using key_type = std::size_t;
using graph_type = YAGL::Graph<key_type, SpatialNode3D<Vertex>>;
using grammar_type = DGGML::Grammar<graph_type>;

template <typename G>
Vertex& vertex(G& g, key_type k) { return g[k].template getData<Vertex>(); }

// ----------------------------------------------------------------------------
// MFEM operator for the Solver: M du/dt = -K u on the interior dofs.
// ----------------------------------------------------------------------------
class HeatFemOperator {

public:
    int ndofs = 0;
    int fem_offset = 0;            // set by the Solver
    std::vector<double> u;         // persistent field (survives Solver rebuilds)

    HeatFemOperator(int num_elements, double length, double kappa, double u_left, double u_right)
        : mesh(mfem::Mesh::MakeCartesian1D(num_elements, length)),
          fec(1, 1),
          fespace(&mesh, &fec)
    {
        ndofs = fespace.GetNDofs();

        mfem::ConstantCoefficient kappa_coeff(kappa);
        mfem::BilinearForm K_form(&fespace);
        K_form.AddDomainIntegrator(new mfem::DiffusionIntegrator(kappa_coeff));
        K_form.Assemble();
        K_form.Finalize();
        K = K_form.SpMat();

        mfem::BilinearForm M_form(&fespace);
        M_form.AddDomainIntegrator(new mfem::MassIntegrator());
        M_form.Assemble();
        M_form.Finalize();
        M = M_form.SpMat();

        // Dirichlet ends: the dofs are held, so remove them from the mass solve
        M.EliminateRowCol(0);
        M.EliminateRowCol(ndofs - 1);

        // K restricted to the interior (ends -> zero rows/cols) for the implicit solve
        K_dir = K;
        K_dir.EliminateRowCol(0, mfem::Operator::DIAG_ZERO);
        K_dir.EliminateRowCol(ndofs - 1, mfem::Operator::DIAG_ZERO);

        A_solver.SetRelTol(1e-10);
        A_solver.SetMaxIter(5000);
        A_solver.SetPrintLevel(0);
        M_prec.SetOperator(M);
        M_solver.SetPreconditioner(M_prec);
        M_solver.SetOperator(M);
        M_solver.SetRelTol(1e-12);
        M_solver.SetMaxIter(200);
        M_solver.SetPrintLevel(0);

        u.assign(ndofs, 0.0);
        u.front() = u_left;
        u.back() = u_right;
    }

    // ---- interface required by DGGML::Solver ----

    void Mult(double /*t*/, N_Vector y, N_Vector ydot)
    {
        mfem::Vector uy(N_VGetArrayPointer(y) + fem_offset, ndofs);
        mfem::Vector Ku(ndofs), du(ndofs);
        K.Mult(uy, Ku);
        Ku.Neg();
        Ku(0) = 0.0;                 // Dirichlet dofs do not move
        Ku(ndofs - 1) = 0.0;
        du = 0.0;
        M_solver.Mult(Ku, du);

        double* ydot_ptr = N_VGetArrayPointer(ydot) + fem_offset;
        for (int i = 0; i < ndofs; ++i)
            ydot_ptr[i] += du(i);
    }

    void Load(N_Vector y)
    {
        std::copy(u.begin(), u.end(), N_VGetArrayPointer(y) + fem_offset);
    }

    void Sync(N_Vector y)
    {
        const double* src = N_VGetArrayPointer(y) + fem_offset;
        std::copy(src, src + ndofs, u.begin());
    }

    // ---- implicit integration (detected by the Solver through PrecSolve) ----
    // Mult() is linear, so the Solver integrates it implicitly. The Newton matrix on the
    // FEM segment is I - gamma*J = I + gamma M^-1 K, so  z = (M + gamma K)^-1 M r.
    // The end dofs are Dirichlet (identity rows).

    void PrecSetup(double gamma)
    {
        if (gamma == prec_gamma) return;
        A.reset(mfem::Add(1.0, M, gamma, K_dir));
        A_prec.reset(new mfem::BlockILU(*A, 1, mfem::BlockILU::Reordering::NONE));
        A_solver.SetPreconditioner(*A_prec);
        A_solver.SetOperator(*A);
        prec_gamma = gamma;
    }

    void PrecSolve(const double* r, double* z, double gamma)
    {
        PrecSetup(gamma);   // no-op unless gamma changed since the last factorization
        mfem::Vector rv(const_cast<double*>(r), ndofs), zv(z, ndofs), b(ndofs);
        M.Mult(rv, b);
        zv = 0.0;
        A_solver.Mult(b, zv);
    }

    // ---- hosting geocell ----
    // The field is integrated in exactly one geocell's solver: the first one
    // that has solving-rule matches claims the operator.
    bool claim(std::size_t k, bool has_rules)
    {
        std::lock_guard<std::mutex> lock(claim_mutex);
        if (owner == npos && has_rules) {
            owner = k;
            std::cout << "[HeatFemOperator] hosted by geocell " << k << "\n";
        }
        return owner == k;
    }

private:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
    std::size_t owner = npos;
    std::mutex claim_mutex;

    mfem::Mesh mesh;
    mfem::H1_FECollection fec;
    mfem::FiniteElementSpace fespace;
    mfem::SparseMatrix K, M;
    mfem::SparseMatrix K_dir;
    mfem::CGSolver M_solver;
    mfem::BlockILU M_prec{1, mfem::BlockILU::Reordering::NONE};   // ILU(0): exact for 1D (tridiagonal)

    std::unique_ptr<mfem::SparseMatrix> A;        // M + gamma K_dir
    std::unique_ptr<mfem::BlockILU> A_prec;
    mfem::CGSolver A_solver;
    double prec_gamma = -1.0;

};


// PDE rule: the vertex temperature follows the MFEM dof of the vertex.
// Mult() has already filled the FEM segment of ydot when the grammar rules run.
void heat_pde_rule(grammar_type& grammar, HeatFemOperator* op)
{
    graph_type lhs, rhs;
    lhs.addNode({1, {Vertex{}}});
    rhs.addNode({1, {Vertex{}}});

    DGGML::SolvingRule<graph_type> rule(
        "heat_pde", lhs, rhs, 1,
        [](auto& g, auto& m, auto& varset) {
            varset.insert(&vertex(g, m[1]).u);
        },
        [op](auto& g, auto& m, auto y, auto ydot, auto& varmap) {
            // Only the solver that carries the FEM segment has a FEM dof to follow
            if (N_VGetLength(ydot) != static_cast<sunindextype>(varmap.size()) + op->ndofs + 1)
                return;
            auto& v = vertex(g, m[1]);
            NV_Ith_S(ydot, varmap[&v.u]) += NV_Ith_S(ydot, op->fem_offset + v.dof);
        });
    grammar.addRule(rule);
}

class HeatRodFemModel : public DGGML::Model3D<grammar_type> {

public:

    using fem_operator_type = HeatFemOperator;

    fem_operator_type* fem_operator(std::size_t k, bool has_rules)
    {
        return (fem && fem->claim(k, has_rules)) ? fem.get() : nullptr;
    }

    void initialize() override
    {
        std::cout << "Initializing MFEM 1D heat equation on a link\n";
        name = "HeatRodMFEM";

        geoplex2D.init(settings.CELL_NX, settings.CELL_NY, settings.CELL_NZ,
                       settings.CELL_DX, settings.CELL_DY, settings.CELL_DZ,
                       false, settings.varepsilon);

        fem = std::make_shared<HeatFemOperator>(settings.num_elements, settings.rod_length,
                                                settings.kappa, settings.u_left, settings.u_right);

        // One vertex per MFEM dof. MakeCartesian1D numbers the vertices (and so the P1 dofs)
        // from left to right, so dof i sits at x = i * h.
        const double cx = 0.5 * settings.CELL_NX * settings.CELL_DX;
        const double cy = 0.5 * settings.CELL_NY * settings.CELL_DY;
        const double cz = 0.5 * settings.CELL_NZ * settings.CELL_DZ;
        x0 = cx - 0.5 * settings.rod_length;

        const int n = fem->ndofs;
        const double h = settings.rod_length / settings.num_elements;

        // Smooth initial profile (the operator starts from the Dirichlet values only)
        for (int i = 0; i < n; ++i)
            fem->u[i] = exact(x0 + i * h, 0.0);

        std::vector<key_type> keys;
        for (int i = 0; i < n; ++i) {
            Vertex v;
            v.dof = i;
            v.u = fem->u[i];
            auto k = gen.get_key();
            // v.fixed is true for the Dirichlet ends, false for the interior

            system_graph.addNode({k, {v, x0 + i * h, cy, cz}});
            keys.push_back(k);
        }

        for (int i = 0; i + 1 < n; ++i)
            system_graph.addEdge(keys[i], keys[i + 1]);

        heat_pde_rule(gamma, fem.get());
    }

    void set_parameters(Parameters params) { settings = params; }

    // Exact solution: every sine mode decays independently,
    //   u(x,t) = steady(x) + sum_n a_n sin(n pi x/L) exp(-kappa (n pi/L)^2 t)
    double exact(double x, double t) const
    {
        const double L = settings.rod_length;
        const double xl = x - x0;
        double v = settings.u_left + (settings.u_right - settings.u_left) * xl / L;
        for (auto& [n, a] : settings.modes) {
            const double kn = n * M_PI / L;
            v += a * std::sin(kn * xl) * std::exp(-settings.kappa * kn * kn * t);
        }
        return v;
    }

    // Writes x, graph u, MFEM u, exact and reports the errors.
    void collect(std::size_t step) override
    {
        struct Row { double x, u_graph, u_fem; int dof; };
        std::vector<Row> rows;
        for (auto& [key, node] : system_graph.getNodeSetRef()) {
            const auto& v = vertex(system_graph, key);
            rows.push_back({node.getData().position[0], v.u, fem->u[v.dof], v.dof});
        }
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.x < b.x; });

        const double t = step * settings.DELTA;
        double err_fem = 0.0, mirror = 0.0, err_x = 0.0;
        const double hmesh = settings.rod_length / settings.num_elements;
        std::filesystem::create_directories("results");
        const std::string fname = "results/heat_step_" + std::to_string(step) + ".csv";
        std::ofstream out(fname);
        out << std::setprecision(10);
        out << "x,u_graph,u_mfem,exact\n";
        for (auto& r : rows) {
            const double ex = exact(x0 + r.dof * hmesh, t);
            out << r.x << "," << r.u_graph << "," << r.u_fem << "," << ex << "\n";
            if (std::abs(r.u_fem - ex) > err_fem) { err_fem = std::abs(r.u_fem - ex); err_x = r.x; }
            mirror = std::max(mirror, std::abs(r.u_graph - r.u_fem));
        }
        std::cout << "[collect] step " << step << " t=" << t << ": " << rows.size()
                  << std::scientific << std::setprecision(3)
                  << " vertices, max |u_mfem - exact|=" << err_fem << " at x=" << std::fixed << err_x << std::scientific
                  << ", max |u_graph - u_mfem|=" << mirror << std::fixed << " -> " << fname << "\n";
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
        std::vector<double> temperature, mfem_u, exact_u;
        for (auto it = snapshot.node_list_begin(); it != snapshot.node_list_end(); ++it) {
            const auto& v = vertex(snapshot, it->first);
            temperature.push_back(v.u);
            mfem_u.push_back(fem->u[v.dof]);
            exact_u.push_back(exact(it->second.getData().position[0], t));
        }
        vtk_writer.set_extra_point_data({{"temperature", temperature},
                                         {"temperature_mfem", mfem_u},
                                         {"exact", exact_u}});
        vtk_writer.save(snapshot, "results/simulation_step_" + std::to_string(step));
        collect(step);
    }

    Parameters settings;
    double x0 = 0.0;
    // shared so that the model copy made by the runner still sees the same operator
    std::shared_ptr<HeatFemOperator> fem;
};

int main(int argc, char* argv[])
{
    const int SIM_DIM = 3;

    HeatRodFemModel model;
    Parameters params;
    if (argc > 1) params.num_elements = std::atoi(argv[1]);   // number of MFEM elements
    model.set_parameters(params);

    DGGML::SimulationRunner<HeatRodFemModel, SIM_DIM> runner(model, true);
    runner.initialize();
    runner.run();

    return 0;
}
