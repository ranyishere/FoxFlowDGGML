// ===========================================================================
//  EveryLBM (DGG-idiomatic variant of grid_check.cpp)
//
//  Same model, same defaults; the EveryRules are rewritten to follow the
//  Every-rule semantics y_j := x_j + V_{r,j}: each update only builds an
//  increment V from the frozen pre-state and adds it (+=); one generic additive
//  commit reconciles it; symmetric-stencil embeddings are one instantiation (no
//  /sym in the commit); all rules are integer multiples n_r of one fundamental
//  clock; structure changes stay in WithRules. See the section comment "The
//  EveryRule(s)" below. (The legacy 5-node local-gather rule with its
//  hand-built node list was dropped.)
// ===========================================================================

// ===========================================================================
//  EveryLBM : a FOCUSED D2Q9 lattice-Boltzmann stream + collision example
//             driven entirely by the new DGGML::EveryRule.
//
//  This demonstrates the `Every` rule: a clock-scheduled, bulk graph rewrite
//
//     {tau_a(i)(x_i) | i in I}_*  ->  {tau_a(j)(y_j) | j in I}_*
//        every (T_r = n_r * T_clock ; p_r)
//        with { y_j := x_j + V_{r,j}([x_i | ...]) | j in I }
//
//  Here it re-creates the lattice-Boltzmann stream/collision update that the
//  FoxFlowDGGMLParallel NeurDE example implemented with a DiscreteSolveRule.
//  Instead of living inside the ODE solver (y / y_next double buffer), the
//  update is expressed as a WithRule-style `with` block and fired at a fixed
//  clock interval by the SSA engine.
//
//  Key property: LBM streaming is explicit transport, so every site must read
//  its neighbours at their OLD (pre-tick) values. The EveryRule firing code
//  guarantees this by inducing ALL matches from the pre-tick graph first
//  (read-only snapshots), applying the collide+stream update to each, and only
//  THEN committing the results -- i.e. simultaneous (Jacobi) semantics, exactly
//  matching y_j := x_j + V([x_i]).
//
//  Setup (deliberately minimal -- NO neural closure, NO adaptive refinement):
//   * 1D SOD shock tube: a chain of D2Q9 lattice sites along x.
//   * A single `Fluid` node type. Consecutive sites are joined by an edge
//     (the streaming adjacency). Interior sites are the update centres; the two
//     end sites have a single neighbour, never become a centre, and therefore
//     stay fixed -- acting as the high/low pressure reservoirs.
//   * One EveryRule ("lbm_stream_collide") whose LHS is the 3-node streaming
//     stencil  nbr(2) -- centre(1) -- nbr(3). Its `with` block collides the
//     centre and its two neighbours locally, then streams the post-collision
//     populations onto the centre.
// ===========================================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <variant>
#include <cstdlib>
#include <omp.h>
#include <atomic>

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


struct Parameters {
    // Sub-DELTA demo: fire the LBM stream/collide EVERY_SUBFIRES times per DELTA
    // window (T_clock = DELTA / EVERY_SUBFIRES). NUM_STEPS windows therefore run
    // NUM_STEPS * EVERY_SUBFIRES LBM steps in total; kept at 200 so results are
    // directly comparable to the once-per-window baseline / Python reference by
    // matching DGG checkpoint j against reference step j * EVERY_SUBFIRES.
    int         EVERY_SUBFIRES = 4;
    std::size_t NUM_STEPS = 50;   // 50 windows * 4 sub-fires = 200 LBM steps
    double TIME_STEP = 0.1;

    // T_clock : one approximate_ssa() window == one EveryRule clock tick.
    float DELTA = 0.4;
    int   MIN_DELTA_STEPS = 1;
    float DELTA_DELTA_T = DELTA / MIN_DELTA_STEPS;
    float DELTA_T_MIN = DELTA_DELTA_T;

    // Serial single-geocell domain: one cell spanning the whole tube.
    std::size_t CELL_NX = 1;
    std::size_t CELL_NY = 1;
    std::size_t CELL_NZ = 1;
    double CELL_DX = 3.0;
    double CELL_DY = 3.0;
    double CELL_DZ = 3.0;

    // 3.5 * varepsilon < CELL_DX/2 must hold.
    double varepsilon = CELL_DX / 30.0;

    // Reaction radius: must reach an immediate lattice neighbour so the 3-node
    // stencil can be matched. Edges constrain adjacency, so a generous radius is
    // safe.
    float MAXIMAL_REACTION_RADIUS = 0.5;

    // --- lattice geometry ---
    int    NUM_SITES    = 3001;   // uniform lattice sites along the tube (case 1)
    double STREAM_LENGTH = (CELL_NX * CELL_DX - 2.0 * varepsilon) / (NUM_SITES - 1);

    // --- D2Q9 thermal closure parameters (analytic reference, no NN) ---
    // Values mirror SOD case 1 in NeurDE Sod_cases_param.yml.
    double vuy    = 2.0;    // Cv = 1/(vuy-1); gamma = vuy/(vuy-1) = 2 in 2D
    double Pr     = 0.71;   // Prandtl number
    double muy    = 0.025;  // dynamic viscosity (sets the BGK relaxation time)
    double alpha1 = 1.08;   // entropy relaxation-time amplifier (case 1)
    double Uax    = 0.057;  // moving-frame advection velocity (case 1)
    double Uay    = 0.0;    // moving-frame advection velocity in y

    // --- adaptive mesh refinement (single-level, clocked-native) ---
    // When AMR is on: a second EveryRule ("lbm_stream_fine") sub-cycles the
    // refined (level-2) sites at half the base clock, and two structural
    // WithRules (amr_refine / amr_coarsen) insert/remove a midpoint site on
    // steep/flat coarse edges. The refine/coarsen criteria are normalised by
    // the peak |drho/dx| tracked at each checkpoint (see g_peak_grad).
    bool   AMR          = false;  // enable refine/coarsen + fine sub-cycle rule
    double REFINE_FRAC  = 0.30;   // refine  when |grad| > REFINE_FRAC  * peak grad
    double COARSEN_FRAC = 0.08;   // coarsen when |grad| < COARSEN_FRAC * peak grad
    double REFINE_FLOOR = 1e-4;   // absolute floor on the refine gradient
    double K_REFINE     = 8.0;    // refine  rate constant (propensity scale)
    double K_COARSEN    = 2.0;    // coarsen rate constant (propensity scale)
    long   MAX_REFINE   = 1000000;// cap on cumulative refine events (diagnostic)
    double REFINE_VISC  = 0.5;    // local dissipation blend for refined sites
    double VISC_FINE    = 1.0;    // (tau-1/2) factor at level 2 (0.5 keeps nu/4 vs coarse)
    // Shock limiter on refined sites: the non-equilibrium sensor eps ramps the
    // (tau-1/2) factor from VISC_FINE (eps<=LIM_E0) to LIM_VISC (eps>=LIM_E1).
    double LIM_E0       = 0.0;    // sensor level where damping starts (LIM_E1<=LIM_E0: off)
    double LIM_E1       = 0.0;    // sensor level of full damping
    double LIM_VISC     = 1.0;    // (tau-1/2) factor at full damping

    bool        WRITE_VTK  = false;
    std::string RESULTS_DIR = "results_sub4";
};

// ---------------------------------------------------------------------------
//  Global AMR state (serial single-geocell run -> plain globals are fine)
// ---------------------------------------------------------------------------
// Peak |drho/dx| measured at the previous checkpoint; normalises the
// refine/coarsen gradient criteria so they track the moving shock front.
static double g_peak_grad    = 1e-6;
// Reference scales + weights for the multi-field refinement indicator (see
// amr_grad). Scales are set from the initial condition in initialize();
// weights are env knobs (SOD_AMR_WT / SOD_AMR_WU; 0 disables that field).
static double g_amr_ref_rho = 1.0;
static double g_amr_ref_T   = 1.0;
static double g_amr_ref_u   = 1.0;
static double g_amr_wT      = 1.0;
static double g_amr_wU      = 1.0;
// Refine factor m (SOD_SPLIT): a refined link is cut into m links, so Shock
// sites sit at level m and Shock2 sites at level m*m. m = 2 is the original
// midpoint scheme. g_seed_level (SOD_SEED_LEVEL) selects Shock (1) or Shock2 (2)
// for the t=0 seeded replace-mode band.
static int    g_split      = 2;
static int    g_seed_level = 1;
static long   g_refine_count = 0;   // cumulative refine  events (diagnostic)
static long   g_coarsen_count = 0;  // cumulative coarsen events (diagnostic)
// Positions (x) of the live refined midpoints. The refine rule keeps refined
// nodes at least ~1.5 STREAM_LENGTH apart so every refined site always has a
// genuine COARSE neighbour (and coarse outer neighbour) on each side; that is
// what the time-interpolated streaming path assumes. Two refined nodes sharing
// a coarse node would leave that coarse node's far neighbour refined, breaking
// the local full-coarse-step reconstruction and diverging at the shock.
static std::vector<double> g_refined_x;

static double env_d(const char *k, double d) { const char *v = std::getenv(k); return v ? std::atof(v) : d; }
static int    env_i(const char *k, int d)    { const char *v = std::getenv(k); return v ? std::atoi(v) : d; }
static bool   env_b(const char *k, bool d)   {
    const char *v = std::getenv(k);
    if (!v) return d;
    return !(v[0] == '0' || v[0] == '\0' || v[0] == 'f' || v[0] == 'F');
}

// ---------------------------------------------------------------------------
//  D2Q9 velocity stencil (x/y components and lattice weights)
// ---------------------------------------------------------------------------
namespace D2Q9 {
    constexpr int    Q     = 9;
    constexpr double ex[Q] = { 1, 0, -1,  0,  1, -1, -1,  1, 0};
    constexpr double ey[Q] = { 0, 1,  0, -1,  1,  1, -1, -1, 0};
}

// The inner lattice carried by every site: the density (f) and energy (g)
// populations plus the macroscopic moments they encode.
struct LatticeData {
    double f[D2Q9::Q];   // density distribution populations f_i
    double g[D2Q9::Q];   // energy  distribution populations g_i
    double rho;
    double ux, uy;
    double T;
    double level;        // AMR level: 1 = coarse (base), 2 = refined (half-spacing)
    double grad_scale;   // smoothed local |drho/dx| envelope (AMR band membership)
    LatticeData() : f{}, g{}, rho(0.0), ux(0.0), uy(0.0), T(0.0), level(1.0), grad_scale(0.0) {}
};

// Site node types. Fluid is the coarse (level-1) lattice cell. Shock/Shock2 are
// the level-2/level-4 refined sites introduced by the AMR refine rules. Left/
// RightBoundary are the domain-end reservoir/open walls. StreamLink is a MEDIATOR
// (not a physical site): it sits on each streaming arm so the streaming pattern
// matches through an explicit link node, letting the AMR rules attach/detach a
// refined arm structurally while the coarse arm is preserved. Mirrors the
// non-clocked NeurDE mediator architecture.
struct Fluid         { LatticeData lat; };
struct LeftBoundary  { LatticeData lat; };
struct RightBoundary { LatticeData lat; };
struct Shock         { LatticeData lat; };
struct Shock2        { LatticeData lat; };
struct StreamLink    { LatticeData lat; };

// Per-type refinement level used for DIRECTIONAL PRUNING of the streaming
// variants: a centre only streams from bracket sites whose level does not exceed
// its own (a coarse site never pulls from a finer one). Fluid/boundaries = 1,
// Shock = 2, Shock2 = 4.
template <class T> struct type_level_t { static int v() { return 1; } };
template <> struct type_level_t<Shock>  { static int v() { return g_split; } };
template <> struct type_level_t<Shock2> { static int v() { return g_split * g_split; } };

using key_type = std::size_t;

// Measured simulation time: the streaming EveryRule firings actually executed
// on two probe sites (a far-field coarse Fluid node and, with SOD_SEED, the
// membrane fine node). The engine fires each probe on its own schedule, so
// these counters read the time the scheduler really advanced rather than a
// nominal step*DELTA formula.
static key_type          g_probe_coarse = static_cast<key_type>(-1);
static key_type          g_probe_fine   = static_cast<key_type>(-1);
static std::atomic<long> g_fires_coarse{0};
static std::atomic<long> g_fires_fine{0};

using graph_type = YAGL::ParallelGraph<
    key_type, SpatialNode3D<Fluid, LeftBoundary, RightBoundary, Shock, Shock2, StreamLink>>;
using CplexGraph3D_t = typename DGGML::ExpandedComplex3D<>::types::graph_type;
using grammar_type = DGGML::Grammar<graph_type>;

// Reach the LatticeData carried by any node, over the multi-type variant.
template <typename NodeData>
inline LatticeData &lat_ref(NodeData &nd) {
    return std::visit([](auto &alt) -> LatticeData & { return alt.lat; }, nd.data);
}

// ---------------------------------------------------------------------------
//  D2Q9 thermal lattice-Boltzmann local operators
// ---------------------------------------------------------------------------

// Temperature-dependent on-lattice weights (compressible/thermal weight set).
inline void get_w(double T, double w[D2Q9::Q]) {
    const double a = (1.0 - T) * T * 0.5;   // axis directions   (i=0..3)
    const double d = T * T * 0.25;          // diagonal          (i=4..7)
    const double r = (1.0 - T) * (1.0 - T); // rest              (i=8)
    for (int i = 0; i < 4; ++i) w[i] = a;
    for (int i = 4; i < 8; ++i) w[i] = d;
    w[8] = r;
}

// Seed a site with the rest-state (u=0) thermal equilibrium for its (rho, T).
inline void set_equilibrium(LatticeData &lat) {
    double w[D2Q9::Q]; get_w(lat.T, w);
    const double E = lat.T;   // rest-state specific energy (Cv = 1 for vuy = 2)
    for (int i = 0; i < D2Q9::Q; ++i) {
        lat.f[i] = w[i] * lat.rho;
        lat.g[i] = w[i] * 2.0 * lat.rho * E;
    }
}

// Recover (rho, ux, uy, T) from the f/g populations. The momentum uses the
// moving-frame velocities e + U (matches the Python SODSolver.get_momentum,
// which builds ex/ey as the raw stencil plus Uax/Uay).
inline void compute_macroscopic(LatticeData &lat, double iCv, double Uax, double Uay) {
    double rho = 0.0, jx = 0.0, jy = 0.0, energy = 0.0;
    for (int i = 0; i < D2Q9::Q; ++i) {
        rho    += lat.f[i];
        jx     += (D2Q9::ex[i] + Uax) * lat.f[i];
        jy     += (D2Q9::ey[i] + Uay) * lat.f[i];
        energy += lat.g[i];
    }
    const double inv = (rho > 1e-12) ? 1.0 / rho : 0.0;
    lat.rho = rho;
    lat.ux  = jx * inv;
    lat.uy  = jy * inv;
    const double E  = 0.5 * energy * inv;
    const double uu = lat.ux * lat.ux + lat.uy * lat.uy;
    lat.T = iCv * (E - 0.5 * uu);
    // Realizability floor: a valid state always has rho, T > 0, so this is a
    // no-op for the uniform-grid (AMR-off) scheme and keeps it bit-identical.
    // It only fires at a sub-cycled coarse/fine AMR interface, where a momentary
    // inconsistency can drive rho or T non-positive and blow up the downstream
    // Newton exp(); clamping keeps the state finite instead of emitting NaN.
    if (!(lat.rho > 1e-12)) lat.rho = 1e-12;
    if (!(lat.T   > 1e-9))  lat.T   = 1e-9;
}

// Multi-field refinement indicator between two sites a, b a distance len apart:
//     max( |d rho|, wT*rho_ref*|dT|/T_ref, wU*rho_ref*|d u|/u_ref ) / len
// The T and velocity gradients are rescaled into density-gradient units, so the
// existing REFINE_FRAC / COARSEN_FRAC / GRAD_FLOOR thresholds and the smoothed
// grad_scale envelope keep their meaning (wT = wU = 0 gives density only).
inline double amr_grad(const LatticeData &a, const LatticeData &b, double len,
                       const Parameters &p) {
    LatticeData ma = a, mb = b;
    compute_macroscopic(ma, p.vuy - 1.0, p.Uax, p.Uay);
    compute_macroscopic(mb, p.vuy - 1.0, p.Uax, p.Uay);
    double g = std::abs(ma.rho - mb.rho);
    if (g_amr_wT > 0.0)
        g = std::max(g, g_amr_wT * g_amr_ref_rho * std::abs(ma.T - mb.T) / g_amr_ref_T);
    if (g_amr_wU > 0.0)
        g = std::max(g, g_amr_wU * g_amr_ref_rho *
                        std::hypot(ma.ux - mb.ux, ma.uy - mb.uy) / g_amr_ref_u);
    return g / std::max(len, 1e-9);
}

// Thermal product-form equilibrium for the density populations f_i. The
// product-form velocity is the peculiar velocity u - U (Python compute_Feq uses
// ux_diff = ux - Uax), while the direction sign pattern uses the raw stencil.
inline void compute_Feq(const LatticeData &lat, double feq[D2Q9::Q],
                        double Uax, double Uay) {
    const double ux = lat.ux - Uax, uy = lat.uy - Uay, T = lat.T;
    auto phi = [T](double u, double e) {
        const double uu = u * u;
        if (e > 0.5)  return (u + uu + T) * 0.5;    // e = +1
        if (e < -0.5) return (-u + uu + T) * 0.5;   // e = -1
        return 1.0 - (uu + T);                       // e =  0
    };
    for (int i = 0; i < D2Q9::Q; ++i)
        feq[i] = lat.rho * phi(ux, D2Q9::ex[i]) * phi(uy, D2Q9::ey[i]);
}

// Exact Levermore energy equilibrium via per-site Newton iteration (a direct
// port of the Python SOD src/getGeq.py levermore_Geq ground-truth solver). It
// finds the three Lagrange multipliers (khi, zetax, zetay) that constrain the
// on-lattice g-distribution to the target energy and energy-flux moments:
//     Sum g            = 2 E            (E  = T*Cv + 1/2 |u|^2)
//     Sum e_x g        = 2 u_x H        (H  = E + T)
//     Sum e_y g        = 2 u_y H
// with g_i = w_i(T) * rho * exp(khi + zetax e_x,i + zetay e_y,i). e_x/e_y are
// the moving-frame velocities (raw stencil + Uax/Uay), matching the Python,
// which passes self.ex = ex1 + Uax into levermore_Geq.
inline void compute_Geq_newton(const LatticeData &lat, double geq[D2Q9::Q],
                               double vuy, double Uax, double Uay) {
    const double Cv = 1.0 / (vuy - 1.0);
    double ux = lat.ux, uy = lat.uy;
    if (std::abs(ux) < 1e-6) ux = 0.0;   // mirror the Python small-value clamps
    if (std::abs(uy) < 1e-6) uy = 0.0;
    const double T = lat.T, rho = lat.rho;
    const double uu = ux * ux + uy * uy;
    const double E  = T * Cv + 0.5 * uu;
    const double H  = E + T;
    double w[D2Q9::Q]; get_w(T, w);
    double ex[D2Q9::Q], ey[D2Q9::Q];
    for (int q = 0; q < D2Q9::Q; ++q) { ex[q] = D2Q9::ex[q] + Uax; ey[q] = D2Q9::ey[q] + Uay; }

    double khi = 0.0, zx = 0.0, zy = 0.0;
    for (int it = 0; it < 50; ++it) {
        double f[D2Q9::Q];
        for (int q = 0; q < D2Q9::Q; ++q)
            f[q] = w[q] * std::exp(khi + zx * ex[q] + zy * ey[q]);

        // residual F and its (symmetric) Jacobian J = dF/d(khi,zx,zy)
        double F0 = 0.0, F1 = 0.0, F2 = 0.0;
        double J00 = 0.0, J01 = 0.0, J02 = 0.0, J11 = 0.0, J12 = 0.0, J22 = 0.0;
        for (int q = 0; q < D2Q9::Q; ++q) {
            const double exq = ex[q], eyq = ey[q], fq = f[q];
            F0 += fq;         F1 += exq * fq;         F2 += eyq * fq;
            J00 += fq;        J01 += exq * fq;        J02 += eyq * fq;
            J11 += exq * exq * fq; J12 += exq * eyq * fq; J22 += eyq * eyq * fq;
        }
        F0 -= 2.0 * E;  F1 -= 2.0 * ux * H;  F2 -= 2.0 * uy * H;

        // solve J d = F for the Newton step (3x3 symmetric, cofactor inverse)
        const double a = J00, b = J01, c = J02, e = J11, ff = J12, i = J22;
        const double A = e * i - ff * ff;
        const double B = ff * c - b * i;
        const double C = b * ff - c * e;
        const double det = a * A + b * B + c * C;

        if (std::abs(det) < 1e-300) break;
        const double invdet = 1.0 / det;
        const double I00 = A * invdet;
        const double I01 = B * invdet;
        const double I02 = C * invdet;
        const double I11 = (a * i - c * c) * invdet;
        const double I12 = (b * c - a * ff) * invdet;
        const double I22 = (a * e - b * b) * invdet;

        const double d0 = I00 * F0 + I01 * F1 + I02 * F2;
        const double d1 = I01 * F0 + I11 * F1 + I12 * F2;
        const double d2 = I02 * F0 + I12 * F1 + I22 * F2;

        khi -= d0;  zx -= d1;  zy -= d2;

        const double mx = std::max(std::abs(d0), std::max(std::abs(d1), std::abs(d2)));
        if (mx < 1e-6) break;
    }

    for (int q = 0; q < D2Q9::Q; ++q)
        geq[q] = w[q] * rho * std::exp(khi + zx * ex[q] + zy * ey[q]);
    // Finiteness guard: for any realizable state the Newton solve converges and
    // this is a no-op (AMR-off stays bit-identical). If a degenerate AMR-
    // interface state makes the multipliers diverge, fall back to the leading
    // (T-weighted) term so the update stays finite instead of poisoning the grid.
    for (int q = 0; q < D2Q9::Q; ++q)
        if (!std::isfinite(geq[q])) { geq[q] = w[q] * rho; }
}

// Dual BGK relaxation rates (density + thermal) with the entropy limiter.
inline void relaxation_rates(const LatticeData &lat, const double feq[D2Q9::Q],
                             double muy, double Pr, double alpha1, double visc,
                             double lim0, double lim1, double limv,
                             double &omega, double &omegaT) {
    // Raw (unclamped) guards to match the NeurDE Python reference exactly.
    const double tau_DL = muy / (lat.rho * lat.T) + 0.5;
    double eps = 0.0;
    for (int i = 0; i < D2Q9::Q; ++i)
        eps += std::abs(lat.f[i] - feq[i]) / feq[i];
    eps /= D2Q9::Q;
    const double alpha = (eps < 1.0) ? alpha1 : (1.0 / tau_DL);
    // shock limiter: damp (raise visc) where the non-equilibrium sensor is large
    if (lim1 > lim0) {
        const double s = std::min(1.0, std::max(0.0, (eps - lim0) / (lim1 - lim0)));
        visc += s * (limv - visc);
    }
    // visc scales (tau - 1/2), i.e. the lattice viscosity (1 on the base level)
    const double tau   = 0.5 + visc * (alpha * tau_DL - 0.5);
    const double tauT  = 0.5 + (tau - 0.5) / Pr;
    omega  = 1.0 / tau;
    omegaT = 1.0 / tauT;
}

// Non-equilibrium heat-flux lattice source Gis (from the pre-collision f). The
// pressure tensor and the projection use the moving-frame velocities e + U to
// match the Python (self.ex2 = self.ex**2, Gis uses self.ex).
inline void compute_Gis(const LatticeData &lat, const double w[D2Q9::Q],
                        double gis[D2Q9::Q], double Uax, double Uay) {
    double Pxx = 0.0, Pyy = 0.0, Pxy = 0.0;
    for (int i = 0; i < D2Q9::Q; ++i) {
        const double exi = D2Q9::ex[i] + Uax, eyi = D2Q9::ey[i] + Uay;
        Pxx += exi * exi * lat.f[i];
        Pyy += eyi * eyi * lat.f[i];
        Pxy += exi * eyi * lat.f[i];
    }
    const double rho = lat.rho, ux = lat.ux, uy = lat.uy, T = lat.T;
    const double Peqxx = rho * ux * ux + rho * T;
    const double Peqyy = rho * uy * uy + rho * T;
    const double Peqxy = rho * ux * uy;
    const double dxy = Pxy - Peqxy;
    const double qsx = 2.0 * ux * (Pxx - Peqxx) + 2.0 * uy * dxy;
    const double qsy = 2.0 * uy * (Pyy - Peqyy) + 2.0 * ux * dxy;
    const double invT = 1.0 / T;  // raw divide to match Python reference
    for (int i = 0; i < D2Q9::Q; ++i)
        gis[i] = w[i] * (qsx * (D2Q9::ex[i] + Uax) + qsy * (D2Q9::ey[i] + Uay)) * invT;
}

// Local BGK collision: relax f toward Feq and g toward Geq plus the heat-flux
// correction. Gis is computed from the PRE-collision f (inside collide_local).
inline void collide(LatticeData &lat, const double feq[D2Q9::Q],
                    const double geq[D2Q9::Q], const double gis[D2Q9::Q],
                    double omega, double omegaT) {
    for (int i = 0; i < D2Q9::Q; ++i) {
        lat.f[i] -= omega * (lat.f[i] - feq[i]);
        lat.g[i] += -omega * (lat.g[i] - geq[i]) + (omega - omegaT) * gis[i];
    }
}

// Full local collision of one lattice from its stored populations.
inline void collide_local(LatticeData &lat, const Parameters &p) {
    compute_macroscopic(lat, p.vuy - 1.0, p.Uax, p.Uay);
    double feq[D2Q9::Q], geq[D2Q9::Q], gis[D2Q9::Q], w[D2Q9::Q];
    get_w(lat.T, w);
    compute_Feq(lat, feq, p.Uax, p.Uay);
    compute_Geq_newton(lat, geq, p.vuy, p.Uax, p.Uay);
    compute_Gis(lat, w, gis, p.Uax, p.Uay);
    double omega, omegaT;
    const double visc = 1.0 + (p.VISC_FINE - 1.0) * (lat.level - 1.0);
    const bool fine = lat.level > 1.5;   // limiter only acts on refined sites
    relaxation_rates(lat, feq, p.muy, p.Pr, p.alpha1, visc,
                     fine ? p.LIM_E0 : 0.0, fine ? p.LIM_E1 : 0.0, p.LIM_VISC,
                     omega, omegaT);
    collide(lat, feq, geq, gis, omega, omegaT);
}

// ---------------------------------------------------------------------------
//  The EveryRule(s) : LBM stream + collision with additive (+=) semantics
// ---------------------------------------------------------------------------
//
//  An Every rule (Tith & Mjolsness, "Clock Rule Semantics for Dynamical Graph
//  Grammars") is the clock-gated bulk rewrite
//
//     {tau_a(i)(x_i) | i in I}_*  ->  {tau_a(j)(y_j) | j in I}_*
//        every (T_r = n_r * T_clock ; p_r)
//        with { y_j := x_j + V_{r,j}([x_i | ...]) | j in I }
//
//  and this file expresses every EveryRule in exactly that form:
//
//   1. INCREMENTS ONLY.  An update never assigns an absolute value. It builds
//      the increment V_{r,j} from the FROZEN pre-state and ADDS it to the RHS
//      copy (apply_increment: y += V). The commit is ONE generic additive
//      reconciliation, live += (after - before), for every additive field
//      (f, g and the grad_scale envelope alike). There is no overwrite and no
//      per-rule division.
//   2. ONE INSTANTIATION PER ORBIT.  "Each fully instantiated update occurs
//      once". A symmetric stencil is matched once per automorphism (mirror
//      image); those embeddings are the SAME instantiation, so only the
//      canonical one (departure bracket A on the left) contributes V and the
//      mirror contributes V = 0. This replaces the old "/sym" correction that
//      hid inside the commit.
//   3. LHS READ-ONLY, DECLARATIVE STENCIL.  The stencil is the LHS pattern
//      (centre + two bracket sites joined through StreamLink mediators); the
//      update reads bound vertices directly, with no locally assembled node
//      containers (no reserve / push_back / sort) and no side channel other
//      than the optional fire probes.
//   4. COMMENSURATE CLOCK.  All rules share ONE fundamental clock T_clock and
//      differ only in the integer multiple n_r (period T_r = n_r * T_clock,
//      phase p_r = 0): Fluid n_r = 4, Shock n_r = 2, Shock2 n_r = 1. Rules due
//      at the same instant form one Jacobi batch whose increments add.
//   5. DATA-ONLY.  Every rule keeps LHS and RHS structurally identical and
//      only changes additive data; type / structure changes (refine, coarsen)
//      stay in the stochastic WithRules. The domain-end reservoirs
//      (Left/RightBoundary) are never a principal node of any rule, i.e. their
//      increment is V = 0 and they remain at their initial (Dirichlet) state.

// The increment V_{r,j} an EveryRule proposes for its principal node.
struct Increment {
    double f[D2Q9::Q] = {};     // density-population increment
    double g[D2Q9::Q] = {};     // energy-population increment
    double grad_scale = 0.0;    // increment of the |drho/dx| envelope
};

// y_j := x_j + V_j.  The RHS copy starts equal to the frozen LHS snapshot x_j,
// so adding V yields exactly the proposed new value.
inline void apply_increment(LatticeData &y, const Increment &V) {
    for (int i = 0; i < D2Q9::Q; ++i) {
        y.f[i] += V.f[i];
        y.g[i] += V.g[i];
    }
    y.grad_scale += V.grad_scale;
}

// The single commit shared by every EveryRule: live += (after - before) = V.
// Increments from overlapping matches therefore accumulate commutatively.
inline DGGML::EveryRule<graph_type>::combine_t additive_combine() {
    return [](graph_type::data_type &live, graph_type::data_type &before,
              graph_type::data_type &after) {
        LatticeData &L = lat_ref(live);
        LatticeData &B = lat_ref(before);
        LatticeData &A = lat_ref(after);
        for (int i = 0; i < D2Q9::Q; ++i) {
            L.f[i] += A.f[i] - B.f[i];
            L.g[i] += A.g[i] - B.g[i];
        }
        L.grad_scale += A.grad_scale - B.grad_scale;
    };
}

// Commensurate clock: one fundamental period, rules are integer multiples.
inline int max_sublevel() { return g_split * g_split; }   // Shock2 sub-cycles m^2 x
inline double fundamental_clock(const Parameters &p) {
    return p.DELTA / (2.0 * max_sublevel());
}
// n_r for a rule whose sites live at `sublevel` (1 coarse, 2 Shock, 4 Shock2).
inline int clock_multiple(int sublevel) { return max_sublevel() / sublevel; }

// ===========================================================================
//  Mediator-based streaming EveryRules.
// ---------------------------------------------------------------------------
//  LHS/RHS is a 5-node mediator path (structure preserving):
//        NbrAT(2) --Ms(4)-- CenterT(1) --Ms(5)-- NbrBT(3)
//  The centre (vertex 1) is the PRINCIPAL node -- the only node the firing
//  engine commits; the two bracket SITES (2,3) and StreamLink MEDIATORS (4,5)
//  are read-only context. Routing every arm through a mediator lets the AMR
//  rules attach/detach a refined arm structurally while the coarse arm is
//  preserved, and lets the streaming pattern be pruned by bracket type
//  (directional level pruning: a coarse centre never streams from a finer one).
//
//  STATIONARY FRAME (u = 0): on a uniform sub-grid the semi-Lagrangian
//  departure lands exactly on the immediate neighbour (weight 1), so streaming
//  is diffusion-free and the coarse/fine interface is exact in x. The advection
//  distance is read off the centre level, s = STREAM_LENGTH / level.
//
//  The increment of the centre is
//        V_f = f_streamed - f_old ,  V_g = g_streamed - g_old     (collide+stream)
//        V_grad = alpha * (|drho/dx|_inst - grad_scale_old)       (EMA envelope)
//  all computed from the frozen snapshot.

// One streaming variant: centre = CenterT, brackets = NbrAT/NbrBT through the
// two StreamLink mediators. Centre = pattern vertex 1 = principal.
template <typename CenterT, typename NbrAT, typename NbrBT>
void add_lbm_stream_link_rule(grammar_type &grammar, const std::string &name,
                              const Parameters &settings, int sublevel) {
    graph_type lhs;
    lhs.addNode({1, {CenterT{}}});     // centre (updated site) = principal
    lhs.addNode({2, {NbrAT{}}});       // one departure bracket (a site)
    lhs.addNode({3, {NbrBT{}}});       // the other departure bracket (a site)
    lhs.addNode({4, {StreamLink{}}});  // mediator on the NbrA arm
    lhs.addNode({5, {StreamLink{}}});  // mediator on the NbrB arm
    lhs.addEdge(2, 4);
    lhs.addEdge(4, 1);
    lhs.addEdge(1, 5);
    lhs.addEdge(5, 3);
    graph_type rhs;
    rhs.addNode({1, {CenterT{}}});
    rhs.addNode({2, {NbrAT{}}});
    rhs.addNode({3, {NbrBT{}}});
    rhs.addNode({4, {StreamLink{}}});
    rhs.addNode({5, {StreamLink{}}});
    rhs.addEdge(2, 4);
    rhs.addEdge(4, 1);
    rhs.addEdge(1, 5);
    rhs.addEdge(5, 3);

    const Parameters p = settings;
    // The path is symmetric under the L<->R reflection (which also swaps the two
    // StreamLinks) only when the two bracket types match: then the matcher emits
    // two embeddings of ONE instantiation and only the canonical one acts.
    constexpr bool symmetric = std::is_same<NbrAT, NbrBT>::value;

    auto update = [p](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto & /*m2*/) {
        const key_type kc = m1[1];
        const key_type kA = m1[2];
        const key_type kB = m1[3];

        const bool aIsLeft = lhs_g[kA].position[0] <= lhs_g[kB].position[0];
        const key_type kL = aIsLeft ? kA : kB;
        const key_type kR = aIsLeft ? kB : kA;

        // fire probes (diagnostic only): count one embedding per firing
        if (aIsLeft) {
            if (kc == g_probe_coarse)    ++g_fires_coarse;
            else if (kc == g_probe_fine) ++g_fires_fine;
        }

        // Mirror embedding of a symmetric stencil: the same instantiation, V = 0.
        if (symmetric && !aIsLeft) return;

        const LatticeData &x = lat_ref(lhs_g[kc]);    // frozen pre-state of the centre
        Increment V;

        // --- gradient companion: EMA of the pre-collision rho envelope ------
        auto rho_pre = [&](key_type k) {
            const LatticeData &l = lat_ref(lhs_g[k]);
            double r = 0.0; for (int i = 0; i < D2Q9::Q; ++i) r += l.f[i]; return r;
        };
        const double xC   = lhs_g[kc].position[0];
        const double rc   = rho_pre(kc);
        double inst = 0.0;
        for (key_type k : {kA, kB}) {
            const double len = std::abs(lhs_g[k].position[0] - xC);
            if (len > 1e-9)
                inst = std::max(inst, p.AMR ? amr_grad(x, lat_ref(lhs_g[k]), len, p)
                                            : std::abs(rc - rho_pre(k)) / len);
        }
        const double alpha = 0.5;
        V.grad_scale = alpha * (inst - x.grad_scale);

        // --- collide C, L, R from the snapshot, then stream onto the centre --
        LatticeData cC = lat_ref(lhs_g[kc]); collide_local(cC, p);
        LatticeData cL = lat_ref(lhs_g[kL]); collide_local(cL, p);
        LatticeData cR = lat_ref(lhs_g[kR]); collide_local(cR, p);

        const double xL = lhs_g[kL].position[0];
        const double xR = lhs_g[kR].position[0];
        const double s  = p.STREAM_LENGTH / x.level;

        double wL = (std::abs(xC - xL) > 1e-12) ? s / (xC - xL) : 0.0;
        double wR = (std::abs(xR - xC) > 1e-12) ? s / (xR - xC) : 0.0;
        if (wL < 0.0) wL = 0.0; if (wL > 1.0) wL = 1.0;
        if (wR < 0.0) wR = 0.0; if (wR > 1.0) wR = 1.0;

        for (int i = 0; i < D2Q9::Q; ++i) {
            double nf, ng;
            if (D2Q9::ex[i] > 0) {            // +x mover departs from the left
                nf = (1.0 - wL) * cC.f[i] + wL * cL.f[i];
                ng = (1.0 - wL) * cC.g[i] + wL * cL.g[i];
            } else if (D2Q9::ex[i] < 0) {     // -x mover departs from the right
                nf = (1.0 - wR) * cC.f[i] + wR * cR.f[i];
                ng = (1.0 - wR) * cC.g[i] + wR * cR.g[i];
            } else {                          // rest population stays on C
                nf = cC.f[i]; ng = cC.g[i];
            }
            V.f[i] = nf - x.f[i];
            V.g[i] = ng - x.g[i];
        }

        apply_increment(lat_ref(rhs_g[kc]), V);     // y := x + V
    };

    DGGML::EveryRule<graph_type> rule(
        name, lhs, rhs, std::move(update),
        /*n_r=*/clock_multiple(sublevel), /*p_r=*/0.0,
        /*T_clock=*/fundamental_clock(p), additive_combine());
    grammar.addRule(rule);
}

// Register all bracket-pair streaming variants for one centre type, pruned so a
// centre only streams from brackets no finer than max_level.
template <typename CenterT>
void add_center_variants_link(grammar_type &grammar, const std::string &cname,
                              const Parameters &settings, int sublevel, int max_level) {
    auto reg = [&](auto A, auto B, const char *suffix) {
        using AT = decltype(A); using BT = decltype(B);
        if (std::max(type_level_t<AT>::v(), type_level_t<BT>::v()) <= max_level)
            add_lbm_stream_link_rule<CenterT, AT, BT>(
                grammar, cname + std::string(suffix), settings, sublevel);
    };
    reg(Fluid{},        Fluid{},         "_F_F");
    reg(Fluid{},        Shock{},         "_F_S");
    reg(Fluid{},        Shock2{},        "_F_S2");
    reg(Shock{},        Shock{},         "_S_S");
    reg(Shock{},        Shock2{},        "_S_S2");
    reg(Shock2{},       Shock2{},        "_S2_S2");
    reg(LeftBoundary{}, Fluid{},         "_L_F");
    reg(LeftBoundary{}, Shock{},         "_L_S");
    reg(LeftBoundary{}, Shock2{},        "_L_S2");
    reg(Fluid{},        RightBoundary{}, "_F_R");
    reg(Shock{},        RightBoundary{}, "_S_R");
    reg(Shock2{},       RightBoundary{}, "_S2_R");
}

// Register the streaming rules. Fluid centre variants always; Shock/Shock2
// centre variants only under AMR (they never match a coarse-only grid).
void add_lbm_streaming_rules(grammar_type &grammar, const Parameters &settings) {
    // Replace-mode: a boundary Fluid legitimately pulls from the end Shock of
    // the fine region across a COARSE link (distance L, weight 1).
    const bool replace = settings.AMR && env_b("SOD_AMR_REPLACE", false);
    const int  m = g_split, m2 = g_split * g_split;
    const bool deep = replace && g_seed_level >= 2;   // seeded Shock2 band
    add_center_variants_link<Fluid>(grammar, "fluid", settings, 1,
                                    replace ? (deep ? m2 : m) : 1);
    if (settings.AMR) {
        add_center_variants_link<Shock>(grammar, "shock", settings, m, m);
        if (!replace || deep)
            add_center_variants_link<Shock2>(grammar, "shock2", settings, m2, m2);
    }
}

// ---------------------------------------------------------------------------
//  AMR : gradient-driven structural refine / coarsen (stochastic WithRules)
// ---------------------------------------------------------------------------
//  refine  : match a steep COARSE edge  A(1) -- B(2)  and insert a midpoint
//            site S(3, level 2)  ->  A -- S -- B.  (edge 1-2 destroyed; nodes
//            1,2 kept; node 3 created; edges 1-3, 3-2 created.)
//  coarsen : match a flat refined cell  A(1) -- S(3) -- B(2)  (S is the level-2
//            articulation) and delete S  ->  A -- B.  (node 3 destroyed; edges
//            1-3, 3-2 destroyed; edge 1-2 created.)
//  Both propensities read LIVE rho from the populations (sum f), since the
//  cached lat.rho is only refreshed at checkpoints, and normalise against the
//  peak |drho/dx| (g_peak_grad). Level guards keep refinement single-level and
//  prevent unbounded recursion.
// ---------------------------------------------------------------------------
//  REPLACE-MODE AMR (SOD_AMR_REPLACE=1): the refined region REPLACES the coarse
//  Fluid sites instead of riding on top of them.
// ---------------------------------------------------------------------------
//  Invariants:  Fluid sites are only ever linked at the coarse spacing L; a
//  Shock (level 2) region is a uniform fine lattice (spacing L/2, links Ms)
//  whose two end Shocks are joined to the neighbouring coarse Fluids by a
//  coarse StreamLink. Refining a coarse link A--M--B inserts a midpoint and
//  RETYPES any Fluid end to Shock (the coarse link is destroyed), so the wave
//  is evolved by the fine lattice only, and the boundary Fluid pulls from the
//  end Shock at distance L (weight 1) while the end Shock pulls from the
//  Fluid with a half-way (w=0.5) interpolation.
// TE = type of the refined sites / retyped ends (Shock, or Shock2 for the deep
// band); R = split factor of the link (default m; m*m for Shock2 growth).
template <typename TA, typename TB, typename TE = Shock>
void add_replace_refine(grammar_type &grammar, const std::string &name,
                        const Parameters &p, double GRAD_FLOOR, int R = g_split) {
    graph_type lhs;
    lhs.addNode({1, {TA{}}});
    lhs.addNode({2, {TB{}}});
    lhs.addNode({3, {StreamLink{}}});     // coarse link (destroyed)
    lhs.addEdge(1, 3);
    lhs.addEdge(3, 2);
    graph_type rhs;
    rhs.addNode({1, {TE{}}});             // retyped (or already) fine end
    rhs.addNode({2, {TE{}}});
    if (R == 2) {
        rhs.addNode({4, {TE{}}});             // new midpoint site
        rhs.addNode({5, {StreamLink{}}});
        rhs.addNode({6, {StreamLink{}}});
        rhs.addEdge(1, 5);
        rhs.addEdge(5, 4);
        rhs.addEdge(4, 6);
        rhs.addEdge(6, 2);
    } else {
        // R-1 new sites (labels 100+q) cut the coarse link into R fine links
        for (int q = 1; q < R; ++q) rhs.addNode({100 + q, {TE{}}});
        for (int q = 0; q < R; ++q) rhs.addNode({200 + q, {StreamLink{}}});
        for (int q = 0; q < R; ++q) {
            const int a = (q == 0)     ? 1 : 100 + q;
            const int b = (q == R - 1) ? 2 : 101 + q;
            rhs.addEdge(a, 200 + q);
            rhs.addEdge(200 + q, b);
        }
    }

    const double MIN_LEN = 0.75 * p.STREAM_LENGTH;
    DGGML::WithRule<graph_type> rule(
        name, lhs, rhs,
        [p, GRAD_FLOOR, MIN_LEN](graph_type &lhs_g, auto &m) -> double {
            LatticeData &a = lat_ref(lhs_g[m[1]]);
            LatticeData &b = lat_ref(lhs_g[m[2]]);
            const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[2]].position[0]);
            if (len < MIN_LEN) return 0.0;                 // already a fine link
            const double grad = amr_grad(a, b, len, p);
            const double scale = std::max(a.grad_scale, b.grad_scale);
            const double thr   = std::max(GRAD_FLOOR, p.REFINE_FRAC * scale);
            if (g_refine_count >= p.MAX_REFINE) return 0.0;
            return (grad > thr) ? p.K_REFINE * grad : 0.0;
        },
        [p, R](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
            ++g_refine_count;
            LatticeData a = lat_ref(lhs_g[m1[1]]);
            LatticeData b = lat_ref(lhs_g[m1[2]]);
            // retyped ends keep their state (setData wiped it); level -> R
            lat_ref(rhs_g[m2[1]]) = a; lat_ref(rhs_g[m2[1]]).level = double(R);
            lat_ref(rhs_g[m2[2]]) = b; lat_ref(rhs_g[m2[2]]).level = double(R);
            const double x1 = lhs_g[m1[1]].position[0], x2 = lhs_g[m1[2]].position[0];
            const double yy = lhs_g[m1[1]].position[1], zz = lhs_g[m1[1]].position[2];
            if (R == 2) {
                LatticeData &mid = lat_ref(rhs_g[m2[4]]);
                for (int i = 0; i < D2Q9::Q; ++i) {
                    mid.f[i] = 0.5 * (a.f[i] + b.f[i]);
                    mid.g[i] = 0.5 * (a.g[i] + b.g[i]);
                }
                compute_macroscopic(mid, p.vuy - 1.0, p.Uax, p.Uay);
                compute_Feq(mid, mid.f, p.Uax, p.Uay);
                compute_Geq_newton(mid, mid.g, p.vuy, p.Uax, p.Uay);
                mid.level      = 2.0;
                mid.grad_scale = std::max(a.grad_scale, b.grad_scale);
                rhs_g[m2[4]].position[0] = 0.5 * (x1 + x2);
                rhs_g[m2[5]].position[0] = 0.75 * x1 + 0.25 * x2;
                rhs_g[m2[6]].position[0] = 0.25 * x1 + 0.75 * x2;
                for (int k : {4, 5, 6}) {
                    rhs_g[m2[k]].position[1] = yy; rhs_g[m2[k]].position[2] = zz;
                }
                return;
            }
            // general split factor: R-1 new sites at x1 + (q/R)(x2-x1)
            for (int q = 1; q < R; ++q) {
                const double t = double(q) / R;
                LatticeData &s = lat_ref(rhs_g[m2[100 + q]]);
                for (int i = 0; i < D2Q9::Q; ++i) {
                    s.f[i] = (1.0 - t) * a.f[i] + t * b.f[i];
                    s.g[i] = (1.0 - t) * a.g[i] + t * b.g[i];
                }
                compute_macroscopic(s, p.vuy - 1.0, p.Uax, p.Uay);
                compute_Feq(s, s.f, p.Uax, p.Uay);
                compute_Geq_newton(s, s.g, p.vuy, p.Uax, p.Uay);
                s.level      = double(R);
                s.grad_scale = std::max(a.grad_scale, b.grad_scale);
                rhs_g[m2[100 + q]].position[0] = (1.0 - t) * x1 + t * x2;
                rhs_g[m2[100 + q]].position[1] = yy;
                rhs_g[m2[100 + q]].position[2] = zz;
            }
            for (int q = 0; q < R; ++q) {
                const double t = (q + 0.5) / R;
                rhs_g[m2[200 + q]].position[0] = (1.0 - t) * x1 + t * x2;
                rhs_g[m2[200 + q]].position[1] = yy;
                rhs_g[m2[200 + q]].position[2] = zz;
            }
        });
    grammar.addRule(rule);
}

void add_amr_replace_rules(grammar_type &grammar, const Parameters &settings) {
    const Parameters p = settings;
    const double GRAD_FLOOR = env_d("SOD_GRAD_FLOOR", 0.05);

    // Matching-cost knobs. refine_ff matches EVERY coarse link in the domain
    // (O(N) propensities per tick); once the front region is seeded only the
    // region ends (refine_fs) need to grow, an O(1) match set.
    //   SOD_STATIC=1          : no structural rules at all (fixed seeded region)
    //   SOD_NO_REFINE_FF=1    : no refine rule that can start a NEW region
    //   SOD_NO_COARSEN_RULES=1: drop the (never-firing) coarsen rules
    if (env_b("SOD_STATIC", false)) return;
    if (g_seed_level >= 2) {
        // deep band (Shock2, spacing L/m^2): it grows only at its ends, where a
        // Shock2 meets a coarse Fluid across a link of length L
        add_replace_refine<Fluid, Shock2, Shock2>(grammar, "refine_fs2", p, GRAD_FLOOR,
                                                  g_split * g_split);
        return;
    }
    if (!env_b("SOD_NO_REFINE_FF", false))
        add_replace_refine<Fluid, Fluid>(grammar, "refine_ff", p, GRAD_FLOOR);
    add_replace_refine<Fluid, Shock>(grammar, "refine_fs", p, GRAD_FLOOR);
    if (!env_b("SOD_NO_REFINE_SS", false))
        add_replace_refine<Shock, Shock>(grammar, "refine_ss", p, GRAD_FLOOR);
    if (env_b("SOD_NO_COARSEN_RULES", false) || g_split != 2) return;

    {   // ---- coarsen (end cell): F0-M0-S1-Ms-Sm-Ms-S2 -> F0-M0-F1-M-S2 ------
        graph_type lhs;
        lhs.addNode({1, {Shock{}}});        // region end (becomes Fluid)
        lhs.addNode({2, {Shock{}}});
        lhs.addNode({3, {Shock{}}});        // midpoint (deleted)
        lhs.addNode({4, {Fluid{}}});        // coarse neighbour of the end
        lhs.addNode({5, {StreamLink{}}});   // coarse link F0 -- S1
        lhs.addNode({6, {StreamLink{}}});
        lhs.addNode({7, {StreamLink{}}});
        lhs.addEdge(4, 5); lhs.addEdge(5, 1);
        lhs.addEdge(1, 6); lhs.addEdge(6, 3);
        lhs.addEdge(3, 7); lhs.addEdge(7, 2);
        graph_type rhs;
        rhs.addNode({1, {Fluid{}}});
        rhs.addNode({2, {Shock{}}});
        rhs.addNode({4, {Fluid{}}});
        rhs.addNode({5, {StreamLink{}}});
        rhs.addNode({8, {StreamLink{}}});   // new coarse link F1 -- S2
        rhs.addEdge(4, 5); rhs.addEdge(5, 1);
        rhs.addEdge(1, 8); rhs.addEdge(8, 2);
        DGGML::WithRule<graph_type> rule(
            "coarsen_end", lhs, rhs,
            [p, GRAD_FLOOR](graph_type &lhs_g, auto &m) -> double {
                LatticeData &a = lat_ref(lhs_g[m[1]]);
                LatticeData &b = lat_ref(lhs_g[m[2]]);
                const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[2]].position[0]);
                const double grad = amr_grad(a, b, len, p);
                const double scale = std::max({a.grad_scale, b.grad_scale, GRAD_FLOOR});
                return (grad < p.COARSEN_FRAC * scale) ? p.K_COARSEN : 0.0;
            },
            [](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
                ++g_coarsen_count;
                LatticeData a = lat_ref(lhs_g[m1[1]]);
                lat_ref(rhs_g[m2[1]]) = a; lat_ref(rhs_g[m2[1]]).level = 1.0;
                const double x1 = lhs_g[m1[1]].position[0], x2 = lhs_g[m1[2]].position[0];
                rhs_g[m2[8]].position[0] = 0.5 * (x1 + x2);
                rhs_g[m2[8]].position[1] = lhs_g[m1[1]].position[1];
                rhs_g[m2[8]].position[2] = lhs_g[m1[1]].position[2];
            });
        grammar.addRule(rule);
    }

    {   // ---- coarsen (isolated): F-M-S-M-F -> F-M-F-M-F --------------------
        graph_type lhs;
        lhs.addNode({1, {Shock{}}});
        lhs.addNode({2, {Fluid{}}});
        lhs.addNode({3, {Fluid{}}});
        lhs.addNode({4, {StreamLink{}}});
        lhs.addNode({5, {StreamLink{}}});
        lhs.addEdge(2, 4); lhs.addEdge(4, 1);
        lhs.addEdge(1, 5); lhs.addEdge(5, 3);
        graph_type rhs;
        rhs.addNode({1, {Fluid{}}});
        rhs.addNode({2, {Fluid{}}});
        rhs.addNode({3, {Fluid{}}});
        rhs.addNode({4, {StreamLink{}}});
        rhs.addNode({5, {StreamLink{}}});
        rhs.addEdge(2, 4); rhs.addEdge(4, 1);
        rhs.addEdge(1, 5); rhs.addEdge(5, 3);
        DGGML::WithRule<graph_type> rule(
            "coarsen_iso", lhs, rhs,
            [p, GRAD_FLOOR](graph_type &lhs_g, auto &m) -> double {
                LatticeData &s = lat_ref(lhs_g[m[1]]);
                double gmax = 0.0;
                for (int k : {2, 3}) {
                    LatticeData &o = lat_ref(lhs_g[m[k]]);
                    const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[k]].position[0]);
                    gmax = std::max(gmax, amr_grad(s, o, len, p));
                }
                const double scale = std::max(s.grad_scale, GRAD_FLOOR);
                return (gmax < p.COARSEN_FRAC * scale) ? p.K_COARSEN : 0.0;
            },
            [](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
                ++g_coarsen_count;
                LatticeData a = lat_ref(lhs_g[m1[1]]);
                lat_ref(rhs_g[m2[1]]) = a; lat_ref(rhs_g[m2[1]]).level = 1.0;
            });
        grammar.addRule(rule);
    }
}

// AMR rules, ported from the non-clocked NeurDE mediator architecture:
//   * refine   : split a coarse F--M--F link with a half-spacing Shock arm,
//                preserving (and marking) the coarse link         [WithRule]
//   * coarsen  : delete a refined arm and clear the coarse mark   [WithRule]
//   * restrict : pull a lagging coarse Fluid (flanked by two refined Shocks)
//                onto the fine envelope                           [EveryRule]
void add_amr_rules(grammar_type &grammar, const Parameters &settings) {
    if (env_b("SOD_AMR_REPLACE", false)) {
        add_amr_replace_rules(grammar, settings);
        return;
    }
    const Parameters p = settings;
    const double GRAD_FLOOR = env_d("SOD_GRAD_FLOOR", 0.05);

    {   // ---- refine: F(1)--M(3)--F(2)  ->  F(1)-Ms(5)-S(4)-Ms(6)-F(2) -------
        //  The coarse mediator M(3) is PRESERVED and MARKED (grad_scale=1) so
        //  the Fluids keep streaming coarse->coarse. The F--Ms--S bracket
        //  variants of the Fluid centre are pruned (max_level=1), so the fine
        //  arm is only pulled by the Shock; destroying M instead leaves the
        //  neighbouring Fluid with NO matching streaming rule (frozen cell).
        //  The refiner skips marked links; the coarsener clears the mark.
        graph_type lhs;
        lhs.addNode({1, {Fluid{}}});
        lhs.addNode({2, {Fluid{}}});
        lhs.addNode({3, {StreamLink{}}});   // coarse streaming link between the fluids
        lhs.addEdge(1, 3);
        lhs.addEdge(3, 2);

        graph_type rhs;
        rhs.addNode({1, {Fluid{}}});
        rhs.addNode({2, {Fluid{}}});
        rhs.addNode({3, {StreamLink{}}});   // preserved coarse link (marked refined)
        rhs.addNode({4, {Shock{}}});        // new half-spacing refined site
        rhs.addNode({5, {StreamLink{}}});   // finer mediator F(1) <-> S
        rhs.addNode({6, {StreamLink{}}});   // finer mediator S <-> F(2)
        rhs.addEdge(1, 3);
        rhs.addEdge(3, 2);
        rhs.addEdge(1, 5);
        rhs.addEdge(5, 4);
        rhs.addEdge(4, 6);
        rhs.addEdge(6, 2);

        const double MIN_LEN = 0.75 * p.STREAM_LENGTH;
        DGGML::WithRule<graph_type> rule(
            "refine_shock_link", lhs, rhs,
            [p, GRAD_FLOOR, MIN_LEN](graph_type &lhs_g, auto &m) -> double {
                if (lat_ref(lhs_g[m[3]]).grad_scale > 0.5) return 0.0;   // already split
                LatticeData &a = lat_ref(lhs_g[m[1]]);
                LatticeData &b = lat_ref(lhs_g[m[2]]);
                const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[2]].position[0]);
                const double grad = amr_grad(a, b, len, p);
                const double scale = std::max(a.grad_scale, b.grad_scale);
                const double thr   = std::max(GRAD_FLOOR, p.REFINE_FRAC * scale);
                if (g_refine_count >= p.MAX_REFINE) return 0.0;
                return (grad > thr && len > MIN_LEN) ? p.K_REFINE * grad : 0.0;
            },
            [p](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
                ++g_refine_count;
                LatticeData &a   = lat_ref(lhs_g[m1[1]]);
                LatticeData &b   = lat_ref(lhs_g[m1[2]]);
                LatticeData &mid = lat_ref(rhs_g[m2[4]]);
                // Seed the refined Shock with the MEAN of its flanking Fluids
                // (conserves the mean moments), recover the mean macro-state,
                // then RESET to the discrete equilibrium of that state. Seeding
                // the averaged NON-equilibrium populations directly leaves a
                // distribution inconsistent with the Newton (Levermore) energy
                // equilibrium, so the fresh site's first collide diverges and
                // blows up (the clocked Geq Newton solve is unforgiving here).
                for (int i = 0; i < D2Q9::Q; ++i) {
                    mid.f[i] = 0.5 * (a.f[i] + b.f[i]);
                    mid.g[i] = 0.5 * (a.g[i] + b.g[i]);
                }
                compute_macroscopic(mid, p.vuy - 1.0, p.Uax, p.Uay);
                compute_Feq(mid, mid.f, p.Uax, p.Uay);
                compute_Geq_newton(mid, mid.g, p.vuy, p.Uax, p.Uay);
                mid.level      = 2.0;
                mid.grad_scale = 1.0;    // seed a high envelope so it holds a step
                lat_ref(rhs_g[m2[3]]).grad_scale = 1.0;   // mark preserved coarse link
                const double x1 = lhs_g[m1[1]].position[0], x2 = lhs_g[m1[2]].position[0];
                const double yy = lhs_g[m1[1]].position[1], zz = lhs_g[m1[1]].position[2];
                rhs_g[m2[4]].position[0] = 0.5 * (x1 + x2);            // Shock at midpoint
                rhs_g[m2[4]].position[1] = yy; rhs_g[m2[4]].position[2] = zz;
                rhs_g[m2[5]].position[0] = 0.75 * x1 + 0.25 * x2;      // F(1)<->S mediator
                rhs_g[m2[5]].position[1] = yy; rhs_g[m2[5]].position[2] = zz;
                rhs_g[m2[6]].position[0] = 0.25 * x1 + 0.75 * x2;      // S<->F(2) mediator
                rhs_g[m2[6]].position[1] = yy; rhs_g[m2[6]].position[2] = zz;
            });
        grammar.addRule(rule);
    }

    {   // ---- coarsen: F(1)-Ms(4)-S(3)-Ms(5)-F(2) -> F(1)-M(6)-F(2) ---------
        //  Inverse of refine: destroy the Shock and its two fine mediators; the
        //  preserved coarse mediator M(6) survives and its mark is cleared.
        graph_type lhs;
        lhs.addNode({1, {Fluid{}}});
        lhs.addNode({2, {Fluid{}}});
        lhs.addNode({3, {Shock{}}});        // refined site to remove
        lhs.addNode({4, {StreamLink{}}});   // F(1) <-> S mediator
        lhs.addNode({5, {StreamLink{}}});   // S <-> F(2) mediator
        lhs.addEdge(1, 4);
        lhs.addEdge(4, 3);
        lhs.addEdge(3, 5);
        lhs.addEdge(5, 2);
        lhs.addNode({6, {StreamLink{}}});   // preserved coarse link
        lhs.addEdge(1, 6);
        lhs.addEdge(6, 2);

        graph_type rhs;
        rhs.addNode({1, {Fluid{}}});
        rhs.addNode({2, {Fluid{}}});
        rhs.addNode({6, {StreamLink{}}});   // coarse link kept (mark cleared)
        rhs.addEdge(1, 6);
        rhs.addEdge(6, 2);

        DGGML::WithRule<graph_type> rule(
            "coarsen_shock_link", lhs, rhs,
            [p, GRAD_FLOOR](graph_type &lhs_g, auto &m) -> double {
                LatticeData &a = lat_ref(lhs_g[m[1]]);
                LatticeData &b = lat_ref(lhs_g[m[2]]);
                const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[2]].position[0]);
                const double grad = amr_grad(a, b, len, p);
                const double scale = std::max({a.grad_scale, b.grad_scale, GRAD_FLOOR});
                return (grad < p.COARSEN_FRAC * scale) ? p.K_COARSEN : 0.0;
            },
            [](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
                ++g_coarsen_count;
                lat_ref(rhs_g[m2[6]]).grad_scale = 0.0;   // re-arm coarse link
            });
        grammar.addRule(rule);
    }

    if (env_b("SOD_AMR2", false)) {
        const double REFINE_FRAC2  = env_d("SOD_REFINE_FRAC2", 0.40);
        const double COARSEN_FRAC2 = env_d("SOD_COARSEN_FRAC2", 0.25);
        const double MIN_LEN2      = 0.375 * p.STREAM_LENGTH;   // only split ~half-spacing arms

        {   // ---- refine level 2: F(1)-Ms(3)-S(2) -> + F(1)-Ms2(5)-S2(4)-Ms2(6)-S(2)
            //  The level-1 arm (F--Ms--S) is PRESERVED and MARKED, and a
            //  quarter-spacing Shock2 arm is added in parallel (same
            //  preserve-and-mark design as level 1). Firing on both arms of a
            //  Shock gives the nested cell F-S2-S-S2-F.
            graph_type lhs;
            lhs.addNode({1, {Fluid{}}});
            lhs.addNode({2, {Shock{}}});
            lhs.addNode({3, {StreamLink{}}});   // level-1 arm link
            lhs.addEdge(1, 3);
            lhs.addEdge(3, 2);

            graph_type rhs;
            rhs.addNode({1, {Fluid{}}});
            rhs.addNode({2, {Shock{}}});
            rhs.addNode({3, {StreamLink{}}});   // preserved level-1 arm (marked)
            rhs.addNode({4, {Shock2{}}});       // new quarter-spacing site
            rhs.addNode({5, {StreamLink{}}});   // F(1) <-> S2
            rhs.addNode({6, {StreamLink{}}});   // S2 <-> S(2)
            rhs.addEdge(1, 3);
            rhs.addEdge(3, 2);
            rhs.addEdge(1, 5);
            rhs.addEdge(5, 4);
            rhs.addEdge(4, 6);
            rhs.addEdge(6, 2);

            DGGML::WithRule<graph_type> rule(
                "refine_shock2_link", lhs, rhs,
                [p, GRAD_FLOOR, REFINE_FRAC2, MIN_LEN2](graph_type &lhs_g, auto &m) -> double {
                    if (lat_ref(lhs_g[m[3]]).grad_scale > 0.5) return 0.0;   // already split
                    LatticeData &a = lat_ref(lhs_g[m[1]]);
                    LatticeData &b = lat_ref(lhs_g[m[2]]);
                    const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[2]].position[0]);
                    const double grad = amr_grad(a, b, len, p);
                    const double scale = std::max(a.grad_scale, b.grad_scale);
                    const double thr   = std::max(GRAD_FLOOR, REFINE_FRAC2 * scale);
                    if (g_refine_count >= p.MAX_REFINE) return 0.0;
                    return (grad > thr && len > MIN_LEN2) ? p.K_REFINE * grad : 0.0;
                },
                [p](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
                    ++g_refine_count;
                    LatticeData &a   = lat_ref(lhs_g[m1[1]]);
                    LatticeData &b   = lat_ref(lhs_g[m1[2]]);
                    LatticeData &mid = lat_ref(rhs_g[m2[4]]);
                    for (int i = 0; i < D2Q9::Q; ++i) {
                        mid.f[i] = 0.5 * (a.f[i] + b.f[i]);
                        mid.g[i] = 0.5 * (a.g[i] + b.g[i]);
                    }
                    compute_macroscopic(mid, p.vuy - 1.0, p.Uax, p.Uay);
                    compute_Feq(mid, mid.f, p.Uax, p.Uay);
                    compute_Geq_newton(mid, mid.g, p.vuy, p.Uax, p.Uay);
                    mid.level      = 4.0;
                    mid.grad_scale = 1.0;
                    lat_ref(rhs_g[m2[3]]).grad_scale = 1.0;   // mark preserved level-1 arm
                    const double xF = lhs_g[m1[1]].position[0], xS = lhs_g[m1[2]].position[0];
                    const double yy = lhs_g[m1[1]].position[1], zz = lhs_g[m1[1]].position[2];
                    rhs_g[m2[4]].position[0] = 0.5 * (xF + xS);            // Shock2 at midpoint
                    rhs_g[m2[4]].position[1] = yy; rhs_g[m2[4]].position[2] = zz;
                    rhs_g[m2[5]].position[0] = 0.75 * xF + 0.25 * xS;      // F(1)<->S2 mediator
                    rhs_g[m2[5]].position[1] = yy; rhs_g[m2[5]].position[2] = zz;
                    rhs_g[m2[6]].position[0] = 0.25 * xF + 0.75 * xS;      // S2<->S(2) mediator
                    rhs_g[m2[6]].position[1] = yy; rhs_g[m2[6]].position[2] = zz;
                });
            grammar.addRule(rule);
        }

        {   // ---- coarsen level 2: remove the quarter-spacing arm, re-arm the level-1 arm
            //  Coarsens ahead of level 1 (COARSEN_FRAC2 > COARSEN_FRAC) so the
            //  nested S2 is gone before its parent Shock is removed.
            graph_type lhs;
            lhs.addNode({1, {Fluid{}}});
            lhs.addNode({2, {Shock{}}});
            lhs.addNode({3, {Shock2{}}});       // site to remove
            lhs.addNode({4, {StreamLink{}}});   // F(1) <-> S2
            lhs.addNode({5, {StreamLink{}}});   // S2 <-> S(2)
            lhs.addNode({6, {StreamLink{}}});   // preserved level-1 arm
            lhs.addEdge(1, 4);
            lhs.addEdge(4, 3);
            lhs.addEdge(3, 5);
            lhs.addEdge(5, 2);
            lhs.addEdge(1, 6);
            lhs.addEdge(6, 2);

            graph_type rhs;
            rhs.addNode({1, {Fluid{}}});
            rhs.addNode({2, {Shock{}}});
            rhs.addNode({6, {StreamLink{}}});
            rhs.addEdge(1, 6);
            rhs.addEdge(6, 2);

            DGGML::WithRule<graph_type> rule(
                "coarsen_shock2_link", lhs, rhs,
                [p, GRAD_FLOOR, COARSEN_FRAC2](graph_type &lhs_g, auto &m) -> double {
                    LatticeData &a = lat_ref(lhs_g[m[1]]);
                    LatticeData &b = lat_ref(lhs_g[m[2]]);
                    const double len = std::abs(lhs_g[m[1]].position[0] - lhs_g[m[2]].position[0]);
                    const double grad = amr_grad(a, b, len, p);
                    const double scale = std::max({a.grad_scale, b.grad_scale, GRAD_FLOOR});
                    return (grad < COARSEN_FRAC2 * scale) ? p.K_COARSEN : 0.0;
                },
                [](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto &m2) {
                    ++g_coarsen_count;
                    lat_ref(rhs_g[m2[6]]).grad_scale = 0.0;   // re-arm level-1 arm
                });
            grammar.addRule(rule);
        }
    }

    {   // ---- restrict: pull the lagging coarse Fluid onto the fine envelope -
        //  Non-clocked pattern S(1)-Ms(4)-F(3)-Ms(5)-S(2) is RENUMBERED so the
        //  Fluid centre is vertex 1 (= the committed principal in the clocked
        //  EveryRule engine). Only the Fluid is pulled; the accurate flanking
        //  Shocks are the fine reference and stay fixed.
        graph_type lhs;
        lhs.addNode({1, {Fluid{}}});        // lagging coarse Fluid (arm centre) = principal
        lhs.addNode({2, {Shock{}}});        // left flanking refined node (accurate)
        lhs.addNode({3, {Shock{}}});        // right flanking refined node (accurate)
        lhs.addNode({4, {StreamLink{}}});   // Sl <-> F mediator
        lhs.addNode({5, {StreamLink{}}});   // F <-> Sr mediator
        lhs.addEdge(2, 4);
        lhs.addEdge(4, 1);
        lhs.addEdge(1, 5);
        lhs.addEdge(5, 3);
        graph_type rhs;
        rhs.addNode({1, {Fluid{}}});
        rhs.addNode({2, {Shock{}}});
        rhs.addNode({3, {Shock{}}});
        rhs.addNode({4, {StreamLink{}}});
        rhs.addNode({5, {StreamLink{}}});
        rhs.addEdge(2, 4);
        rhs.addEdge(4, 1);
        rhs.addEdge(1, 5);
        rhs.addEdge(5, 3);

        const double KAPPA = env_d("SOD_RESTRICT_KAPPA", 0.25);
        // |Aut(LHS)|: the Sl<->Sr reflection (both Shock) swaps the two mediators,
        // so the matcher emits two embeddings of ONE instantiation; only the
        // canonical one (vertex 2 on the left) contributes the increment.
        auto update = [KAPPA](graph_type &lhs_g, graph_type &rhs_g, auto &m1, auto & /*m2*/) {
            const key_type kF  = m1[1];
            const key_type kSl = m1[2];
            const key_type kSr = m1[3];
            if (lhs_g[kSl].position[0] > lhs_g[kSr].position[0]) return;   // mirror: V = 0
            const LatticeData &F  = lat_ref(lhs_g[kF]);
            const LatticeData &Sl = lat_ref(lhs_g[kSl]);
            const LatticeData &Sr = lat_ref(lhs_g[kSr]);
            // V = KAPPA * (mean_S - F)   (the non-clocked KAPPA damping)
            Increment V;
            for (int i = 0; i < D2Q9::Q; ++i) {
                V.f[i] = KAPPA * (0.5 * (Sl.f[i] + Sr.f[i]) - F.f[i]);
                V.g[i] = KAPPA * (0.5 * (Sl.g[i] + Sr.g[i]) - F.g[i]);
            }
            apply_increment(lat_ref(rhs_g[kF]), V);     // y := x + V
        };
        DGGML::EveryRule<graph_type> rule(
            "restrict_shock_link", lhs, rhs, std::move(update),
            /*n_r=*/clock_multiple(1), /*p_r=*/0.0,       // coarse (Fluid) cadence
            /*T_clock=*/fundamental_clock(p), additive_combine());
        grammar.addRule(rule);
    }
}

// ---------------------------------------------------------------------------
//  Model : builds the SOD shock tube and registers the EveryRule
// ---------------------------------------------------------------------------
class LbmModel : public DGGML::Model3D<grammar_type> {

public:

    Parameters settings;

    void set_parameters(Parameters params) { settings = params; }

    void initialize() override {
        std::cout << "Initializing D2Q9 SOD shock tube driven by an EveryRule\n";
        std::cout << "  NUM_SITES=" << settings.NUM_SITES
                  << "  NUM_STEPS=" << settings.NUM_STEPS
                  << "  AMR=" << (settings.AMR ? "ON" : "off") << "\n";
        name = "EveryLBM_SOD";
        g_split      = std::max(2, env_i("SOD_SPLIT", 2));
        g_seed_level = std::max(1, std::min(2, env_i("SOD_SEED_LEVEL", 1)));
        if ((g_split != 2 || g_seed_level != 1) && settings.AMR &&
            !env_b("SOD_AMR_REPLACE", false)) {
            std::cerr << "SOD_SPLIT / SOD_SEED_LEVEL need SOD_AMR_REPLACE=1\n";
            std::exit(1);
        }
        // seeded band refinement factor (m for level 1, m^2 for level 2)
        const int seedR = (g_seed_level >= 2) ? g_split * g_split : g_split;
        std::cout << "  SPLIT=" << g_split << "  SEED_LEVEL=" << g_seed_level
                  << "  seedR=" << seedR << "\n";

        geoplex2D.init(settings.CELL_NX, settings.CELL_NY, settings.CELL_NZ,
                       settings.CELL_DX, settings.CELL_DY, settings.CELL_DZ,
                       false, settings.varepsilon);

        const double margin = settings.varepsilon;
        const double x0 = margin;
        const double x1 = settings.CELL_NX * settings.CELL_DX - margin;
        const double yc = 0.5 * settings.CELL_NY * settings.CELL_DY;
        const double zc = 0.5 * settings.CELL_NZ * settings.CELL_DZ;

        const int    X  = settings.NUM_SITES;
        const double dx = (x1 - x0) / (X - 1);

        // SOD case 1 initial discontinuity (NeurDE case_1_initial_conditions):
        // the split is at index X/2 inclusive on the left. Note case 1 has the
        // LOW-density state on the left and the HIGH-density state on the right.
        double rho_L = 0.5, T_L = 0.2;      // left  (i <= X/2)
        double rho_R = 2.0, T_R = 0.025;    // right (i >  X/2)
        // SOD_CASE2=1: match the non-clocked NeurDE (DiscreteSolveRule) IC and
        // membrane split (case 2: 1.0/0.2 | 0.125/0.16, left = i < X/2).
        const bool case2 = env_b("SOD_CASE2", false);
        if (case2) { rho_L = 1.0; T_L = 0.2; rho_R = 0.125; T_R = 0.16; }
        // Reference scales for the multi-field AMR indicator (amr_grad):
        // full IC range of rho and T, isothermal sound speed for velocity.
        g_amr_ref_rho = std::max(rho_L, rho_R);
        g_amr_ref_T   = std::max(T_L, T_R);
        g_amr_ref_u   = std::sqrt(std::max(T_L, T_R));
        g_amr_wT      = env_d("SOD_AMR_WT", 1.0);
        g_amr_wU      = env_d("SOD_AMR_WU", 1.0);
        // SOD_PY_SPLIT=1: put the membrane where the Python SODSolver does
        // (left = i <= X/2, i.e. one cell right of the non-clocked case-2 split).
        const bool lowSplit = case2 && !env_b("SOD_PY_SPLIT", false);

        // SOD_SEED=W (replace-mode AMR only): start with the W coarse cells on
        // each side of the membrane already refined, so the initial jump never
        // evolves on the coarse lattice while the reactive refiner catches up.
        const int seedW = (settings.AMR && env_b("SOD_AMR_REPLACE", false))
                              ? env_i("SOD_SEED", 0) : 0;
        const int seedMid = X / 2;
        auto in_seed = [&](int i) {
            return seedW > 0 && i >= 1 && i <= X - 2 &&
                   i >= seedMid - seedW && i <= seedMid + seedW - 1;
        };

        std::vector<key_type> ids(X);
        std::vector<double>   xs(X);
        std::vector<LatticeData> lats(X);
        for (int i = 0; i < X; ++i) {
            const double x    = x0 + i * dx;
            const bool   left = lowSplit ? (i < X / 2) : (i <= X / 2);
            LatticeData lat;
            lat.rho = left ? rho_L : rho_R;
            lat.T   = left ? T_L   : T_R;
            lat.ux  = 0.0;
            lat.uy  = 0.0;
            // Seed populations from the product-form Feq and the Levermore
            // Newton Geq (matches the Python case_1_initial_conditions, which
            // uses get_Feq + get_Geq_Newton_solver, not the rest-state weights).
            compute_Feq(lat, lat.f, settings.Uax, settings.Uay);
            compute_Geq_newton(lat, lat.g, settings.vuy, settings.Uax, settings.Uay);
            const key_type key = gen.get_key();
            // Domain-end sites are held reservoir walls (never a streaming
            // centre, so their populations stay frozen at the IC = Dirichlet).
            if (i == 0)
                system_graph.addNode({key, {LeftBoundary{lat}, x, yc, zc}});
            else if (i == X - 1)
                system_graph.addNode({key, {RightBoundary{lat}, x, yc, zc}});
            else if (in_seed(i)) {
                lat.level = double(seedR);
                if (g_seed_level >= 2)
                    system_graph.addNode({key, {Shock2{lat}, x, yc, zc}});
                else
                    system_graph.addNode({key, {Shock{lat}, x, yc, zc}});
            } else
                system_graph.addNode({key, {Fluid{lat}, x, yc, zc}});
            ids[i] = key;
            xs[i]  = x;
            lats[i] = lat;
        }

        // Route every streaming adjacency through a StreamLink MEDIATOR node so
        // the AMR rules can attach/detach refined arms structurally and the
        // streaming pattern can be pruned by bracket type.
        for (int i = 0; i + 1 < X; ++i) {
            if (in_seed(i) && in_seed(i + 1)) {
                // fine link: F -- Ms -- S -- Ms -- ... -- S -- Ms -- F at spacing L/seedR
                const int R = seedR;
                const double xa = xs[i], xb = xs[i + 1];
                auto posf = [&](double t) {
                    if (R == 2) {
                        if (t == 0.5)  return 0.5 * (xa + xb);
                        if (t == 0.25) return 0.75 * xa + 0.25 * xb;
                        if (t == 0.75) return 0.25 * xa + 0.75 * xb;
                    }
                    return (1.0 - t) * xa + t * xb;
                };
                std::vector<key_type> skeys(R), mkeys(R);
                std::vector<LatticeData> mids(R);
                for (int q = 1; q < R; ++q) {
                    const double t = double(q) / R;
                    LatticeData mid = lats[i];
                    for (int k = 0; k < D2Q9::Q; ++k) {
                        mid.f[k] = (1.0 - t) * lats[i].f[k] + t * lats[i + 1].f[k];
                        mid.g[k] = (1.0 - t) * lats[i].g[k] + t * lats[i + 1].g[k];
                    }
                    if (R == 2)
                        for (int k = 0; k < D2Q9::Q; ++k) {
                            mid.f[k] = 0.5 * (lats[i].f[k] + lats[i + 1].f[k]);
                            mid.g[k] = 0.5 * (lats[i].g[k] + lats[i + 1].g[k]);
                        }
                    compute_macroscopic(mid, settings.vuy - 1.0, settings.Uax, settings.Uay);
                    if (env_b("SOD_SEED_EXACT", true)) {
                        // site q of coarse link i is fine index R*i+q of the equivalent
                        // uniform (R*(X-1)+1)-site grid: give it that grid's exact IC
                        // side instead of the smoothed average across the membrane.
                        const int X2 = R * (X - 1) + 1, j = R * i + q;
                        const bool lj = lowSplit ? (j < X2 / 2) : (j <= X2 / 2);
                        mid.rho = lj ? rho_L : rho_R;
                        mid.T   = lj ? T_L   : T_R;
                        mid.ux = 0.0; mid.uy = 0.0;
                    }
                    compute_Feq(mid, mid.f, settings.Uax, settings.Uay);
                    compute_Geq_newton(mid, mid.g, settings.vuy, settings.Uax, settings.Uay);
                    mid.level = double(R);
                    mids[q] = mid;
                    skeys[q] = gen.get_key();
                    if (q == 1 && i == seedMid && g_probe_fine == static_cast<key_type>(-1))
                        g_probe_fine = skeys[q];
                }
                for (int q = 0; q < R; ++q) mkeys[q] = gen.get_key();

                LatticeData none;
                for (int q = 1; q < R; ++q) {
                    if (g_seed_level >= 2)
                        system_graph.addNode({skeys[q], {Shock2{mids[q]}, posf(double(q) / R), yc, zc}});
                    else
                        system_graph.addNode({skeys[q], {Shock{mids[q]},  posf(double(q) / R), yc, zc}});
                }
                for (int q = 0; q < R; ++q)
                    system_graph.addNode({mkeys[q], {StreamLink{none}, posf((q + 0.5) / R), yc, zc}});
                for (int q = 0; q < R; ++q) {
                    const key_type a = (q == 0)     ? ids[i]     : skeys[q];
                    const key_type b = (q == R - 1) ? ids[i + 1] : skeys[q + 1];
                    system_graph.addEdge(a, mkeys[q]);
                    system_graph.addEdge(mkeys[q], b);
                }
                continue;
            }

            LatticeData mlat;   // mediator carries no physics
            const key_type mkey = gen.get_key();
            system_graph.addNode({mkey, {StreamLink{mlat},
                                         0.5 * (xs[i] + xs[i + 1]), yc, zc}});
            system_graph.addEdge(ids[i], mkey);
            system_graph.addEdge(mkey, ids[i + 1]);

        }

        g_probe_coarse = ids[X / 4];
        g_fires_coarse = 0;
        g_fires_fine   = 0;

        add_lbm_streaming_rules(gamma, settings);
        if (settings.AMR) add_amr_rules(gamma, settings);
    }

    void checkpoint(std::size_t step) override {
        // Position-sorted macroscopic profile (x, rho, ux, T) each checkpoint.
        struct Row { double x, rho, ux, T; };
        std::vector<Row> rows;
        std::size_t refined = 0;
        for (auto it = system_graph.node_list_begin(); it != system_graph.node_list_end(); ++it) {
            auto &nd = it->second.getData();
            if (std::holds_alternative<StreamLink>(nd.data)) continue;   // mediators aren't sites
            LatticeData lat = lat_ref(nd);
            if (lat.level >= 1.5) ++refined;
            compute_macroscopic(lat, settings.vuy - 1.0, settings.Uax, settings.Uay);
            rows.push_back({nd.position[0], lat.rho, lat.ux, lat.T});
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row &a, const Row &b) { return a.x < b.x; });

        // Track the peak |drho/dx| so the AMR refine/coarsen criteria can
        // normalise against the current shock strength.
        double peak = 1e-6;
        for (std::size_t i = 1; i < rows.size(); ++i) {
            const double dxi = rows[i].x - rows[i - 1].x;
            if (dxi > 1e-12)
                peak = std::max(peak, std::abs(rows[i].rho - rows[i - 1].rho) / dxi);
        }
        g_peak_grad = peak;

        // Real simulation time exposed per checkpoint:
        //  t_clock = solver's own sampling clock (step * DELTA), same for every
        //            resolution -- the nominal "LBM lattice time".
        //  t_phys  = nominal advection time = 2 coarse firings per window *
        //            step / (NUM_SITES-1) (tube-length units, Python t = iter/(X-1)).
        //            t_lat (below) is the MEASURED counterpart from executed firings.
        //            Written to a times.csv sidecar so the profile
        //            CSV loaders (skiprows=1, 4 cols) stay untouched.
        const double t_clock = static_cast<double>(step) * settings.DELTA;

        const double t_phys  = (settings.NUM_SITES > 1)
                             ? 2.0 * static_cast<double>(step) / (settings.NUM_SITES - 1)
                             : 0.0;

        double rmin = 1e300, rmax = -1e300;

        for (const auto &r : rows) { rmin = std::min(rmin, r.rho); rmax = std::max(rmax, r.rho); }
        std::cout << "  [step " << step << "] sites=" << rows.size()
                  << "  t_clock=" << std::fixed << std::setprecision(4) << t_clock
                  << "  t_phys=" << std::setprecision(6) << t_phys
                  << "  rho[min=" << std::setprecision(4) << rmin
                  << ", max=" << rmax << "]";
        if (settings.AMR)
            std::cout << "  refined=" << refined
                      << "  peakgrad=" << std::setprecision(3) << peak
                      << "  (nRef=" << g_refine_count << " nCoa=" << g_coarsen_count << ")";
        // Measured lattice time (tube-length units, as the Python t = iter/(X-1)):
        // each coarse firing advects STREAM_LENGTH, each fine firing half of it.
        const double Ltube = settings.CELL_NX * settings.CELL_DX - 2.0 * settings.varepsilon;
        const long   nfc   = g_fires_coarse.load();
        const long   nff   = g_fires_fine.load();
        const double t_lat = nfc * settings.STREAM_LENGTH / Ltube;
        const double t_lat_fine = nff * 0.5 * settings.STREAM_LENGTH / Ltube;
        std::cout << "  fires[c=" << nfc << " f=" << nff << "] t_lat="
                  << std::setprecision(6) << t_lat;
        std::cout << "\n";

        std::filesystem::create_directories(settings.RESULTS_DIR);
        std::string fn = settings.RESULTS_DIR + "/sod_step" + std::to_string(step) + ".csv";
        std::ofstream csv(fn);
        csv << "x,rho,ux,T\n" << std::setprecision(10);
        for (const auto &r : rows)
            csv << r.x << "," << r.rho << "," << r.ux << "," << r.T << "\n";

        // Per-checkpoint sim-time sidecar (step,t_clock,t_phys,sites,refined).
        // Truncate + write header at step 0, append afterwards, so a full run
        // leaves one row per checkpoint in RESULTS_DIR/times.csv.
        const std::string tfn = settings.RESULTS_DIR + "/times.csv";
        std::ofstream tcsv(tfn, (step == 0) ? std::ios::trunc : std::ios::app);
        if (step == 0) tcsv << "step,t_clock,t_phys,sites,refined,n_fire_coarse,n_fire_fine,t_lat,t_lat_fine\n";
        tcsv << step << "," << std::setprecision(10) << t_clock << "," << t_phys
             << "," << rows.size() << "," << refined
             << "," << nfc << "," << nff << "," << t_lat << "," << t_lat_fine << "\n";
    }

};

int main(int /*argc*/, char* /*argv*/[]) {
    Parameters params;

    // Environment overrides (default behaviour unchanged when unset).
    params.NUM_SITES = env_i("SOD_NUM_SITES", params.NUM_SITES);
    params.NUM_STEPS = static_cast<std::size_t>(env_i("SOD_NUM_STEPS", static_cast<int>(params.NUM_STEPS)));
    params.STREAM_LENGTH =
        (params.CELL_NX * params.CELL_DX - 2.0 * params.varepsilon) / (params.NUM_SITES - 1);
    params.AMR          = env_b("SOD_AMR",          params.AMR);
    params.Uax          = env_d("SOD_UAX",          params.Uax);
    params.Uay          = env_d("SOD_UAY",          params.Uay);
    params.alpha1       = env_d("SOD_ALPHA1",       params.alpha1);
    params.REFINE_FRAC  = env_d("SOD_REFINE_FRAC",  params.REFINE_FRAC);
    params.COARSEN_FRAC = env_d("SOD_COARSEN_FRAC", params.COARSEN_FRAC);
    params.REFINE_FLOOR = env_d("SOD_REFINE_FLOOR", params.REFINE_FLOOR);
    params.K_REFINE     = env_d("SOD_K_REFINE",     params.K_REFINE);    params.K_COARSEN    = env_d("SOD_K_COARSEN",    params.K_COARSEN);
    params.MAX_REFINE   = env_i("SOD_MAX_REFINE",   static_cast<int>(params.MAX_REFINE));
    params.REFINE_VISC  = env_d("SOD_REFINE_VISC",  params.REFINE_VISC);
    params.VISC_FINE    = env_d("SOD_VISC_FINE",    params.VISC_FINE);
    params.LIM_E0       = env_d("SOD_LIM_E0",       params.LIM_E0);
    params.LIM_E1       = env_d("SOD_LIM_E1",       params.LIM_E1);
    params.LIM_VISC     = env_d("SOD_LIM_VISC",     params.LIM_VISC);

    if (const char *rd = std::getenv("SOD_RESULTS_DIR")) params.RESULTS_DIR = rd;

    LbmModel model;
    model.set_parameters(params);

    DGGML::set_global_seed(12345u);

    DGGML::SimulationRunner<LbmModel, 3> runner(model, true);

    auto t0 = std::chrono::high_resolution_clock::now();
    runner.initialize();
    auto t_run0 = std::chrono::high_resolution_clock::now();
    runner.run();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "run() took "
              << std::chrono::duration<double, std::milli>(t1 - t_run0).count() << " ms\n";

    std::cout << "EveryLBM finished in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    return 0;
}
