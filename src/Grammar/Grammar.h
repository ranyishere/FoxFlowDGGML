#ifndef DGGML_GRAMMAR_H
#define DGGML_GRAMMAR_H

#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <nvector/nvector_serial.h>

namespace DGGML {

    template <typename GraphType>
    struct RuleBase
    {
        using key_type = typename GraphType::key_type;
        std::string rname;

        GraphType lhs_graph;
        GraphType rhs_graph;

        //TODO: figure out why I need the base constructor and can't delete
        //RuleBase() = delete;
        //RuleBase() = default;

        explicit RuleBase(std::string rname, GraphType& lhs_graph, GraphType& rhs_graph) :
        rname(rname), lhs_graph(lhs_graph), rhs_graph(rhs_graph) {}
    };

    template <typename GraphType>
    struct WithRule : RuleBase<GraphType>
    {
        using GraphMapType = std::map<typename GraphType::key_type, typename GraphType::key_type>;
        using propensity_t = std::function<double(GraphType& lhs, GraphMapType& m)>;
        using update_t =  std::function<void(GraphType& lhs, GraphType& rhs, GraphMapType& m1, GraphMapType& m2)>;

        propensity_t propensity;
        update_t update;

        //TODO: figure out why I need the base constructor and can't delete
        //WithRule() = delete;
        //WithRule() {};

        explicit WithRule(std::string rname, GraphType& lhs_graph, GraphType& rhs_graph, propensity_t&& p, update_t&& u)
            : RuleBase<GraphType>(rname, lhs_graph, rhs_graph), propensity(p), update(u) {}

    };

    // ============================================================
    // EveryRule: a clock-scheduled, bulk (Jacobi) graph-rewrite rule.
    //
    // Realizes the operator (clock-node / Pulse form)
    //   {clock(tau), tau_a(i)(x_i) | i in I}_*  ->  {clock(tau), tau_a(j)(y_j) | j in I}_*
    //     with  Pulse(tau mod n_r T_clock - 2 p_r / Lambda ; 1 / Lambda)
    //            * prod_j delta( y_j - x_j - V_{r,j}([x_i | ...]) )
    //   together with one shared clock rule
    //     {clock(tau)} -> {clock(tau)}   solving  dtau/dt = 1 .
    //
    // Semantics:
    //  * Reuses the WithRule update callback signature exactly:
    //      void(GraphType& lhs, GraphType& rhs, GraphMapType& m1, GraphMapType& m2)
    //    where `lhs` is a read-only pre-window snapshot of the match and `rhs`
    //    is the copy whose data is written back.
    //  * The shared clock tau is a genuine integral of dtau/dt = 1: it accumulates
    //    the ACTUAL physical time advanced by each approximate_ssa() window (option
    //    B), so the schedule stays correct even when windows advance by different
    //    amounts of simulation time, and T_clock can be an independent physical
    //    interval (T_clock <= 0 falls back to settings.DELTA). In the Lambda -> inf
    //    limit the Pulse collapses to an exact once-per-period gate: the rule fires
    //    when tau reaches its next scheduled time (start p_r * P, step P = n_r * T_clock).
    //  * Fired FIRST (before the SSA/ODE window), so the gate is evaluated on the
    //    clock value entering the window and the stream/collide is applied before
    //    this window's reactions consume it.
    //  * When it fires, ALL current matches are induced FIRST from the current
    //    graph (read-only snapshots), the update is applied to each, and only
    //    then are the results committed. This gives the bulk `{}_*` rewrite a
    //    provably Jacobi (simultaneous) semantics: neighbours are always read at
    //    their old values, matching y_j := x_j + V([x_i]). If the shared clock
    //    has crossed several periods P in one window, the rule fires once per
    //    crossed boundary as sequential sub-steps (re-inducing between fires).
    //  * The rule is data-only (LHS == RHS structurally). By convention the
    //    update writes the data of the match's PRINCIPAL node (the LHS vertex
    //    with the smallest id, e.g. the stencil centre); neighbour nodes are
    //    read-only context. Only the principal node's data is committed. Commit
    //    is ADDITIVE (live := live + (after - before), realizing y_j := x_j + V)
    //    when the rule supplies a `combine` hook; otherwise the principal is
    //    overwritten. Either way overlapping stencils stay conflict-free.
    // ============================================================
    template <typename GraphType>
    struct EveryRule : RuleBase<GraphType>
    {
        using GraphMapType = std::map<typename GraphType::key_type, typename GraphType::key_type>;
        using update_t = std::function<void(GraphType& lhs, GraphType& rhs, GraphMapType& m1, GraphMapType& m2)>;
        // Optional accumulate/combine hook giving the commit ADDITIVE semantics
        // (y_j := x_j + V_{r,j}). Called as combine(live, before, after) where
        //   before = pre-fire snapshot value x_j,  after = update output x_j + V,
        //   live   = the principal node's current value in the system graph.
        // Empty => overwrite the principal (live := after).
        using combine_t = std::function<void(typename GraphType::data_type& live,
                                             typename GraphType::data_type& before,
                                             typename GraphType::data_type& after)>;

        update_t update;
        combine_t combine; // optional; empty => overwrite commit, else additive
        int n_r;         // clock multiple: period T_r = n_r * T_clock
        double p_r;      // phase offset as a fraction of the period, in [0, 1)
        double T_clock;  // base clock period (physical time); <= 0 => use settings.DELTA

        explicit EveryRule(std::string rname, GraphType& lhs_graph, GraphType& rhs_graph,
                           update_t&& u, int n_r = 1, double p_r = 0.0, double T_clock = 0.0,
                           combine_t&& c = {})
            : RuleBase<GraphType>(rname, lhs_graph, rhs_graph), update(u), combine(c),
              n_r(n_r), p_r(p_r), T_clock(T_clock) {}
    };

    template <typename GraphType>
    struct SolvingRule : RuleBase<GraphType>
    {
        using GraphMapType = std::map<typename GraphType::key_type, typename GraphType::key_type>;
        using initial_condition_t = std::function<void(GraphType& lhs, GraphMapType& m1, std::set<double*>& varset)>;
        using solving_t = std::function<void(GraphType& lhs, GraphMapType& m1, N_Vector y, N_Vector ydot,
                std::unordered_map<double*, std::size_t>& varmap)>;

        initial_condition_t ic;
        solving_t ode;
        std::size_t num_eq;

        explicit SolvingRule(std::string rname, GraphType& lhs_graph, GraphType& rhs_graph,
                             std::size_t num_eq, initial_condition_t&& ic, solving_t&& ode)
                : RuleBase<GraphType>(rname, lhs_graph, rhs_graph), num_eq(num_eq), ic(ic), ode(ode) {}
    };

    //TODO: we need to make sure there is a numbered graph class for rule definitions
    // the numbered graphs has a numbering that is the maximum size of the left and right
    // hand side. All integer labels must differ from each other and they should cover the
    // numbering space and have no gaps. We also need some helper functions. One that
    // returns input only’s = destroyed, shared = maybe changed params not created or
    // destroyed, and the output helper for ones that are only created
    template<typename GraphType>
    struct Grammar
    {
        using graph_type = GraphType;
        using key_type = typename GraphType::key_type;

        std::map<std::string, WithRule<GraphType>> stochastic_rules;
        std::map<std::string, SolvingRule<GraphType>> deterministic_rules;
        std::map<std::string, EveryRule<GraphType>> every_rules;

        void addRule(WithRule<GraphType>& r)
        {
            if(stochastic_rules.find(r.rname) == stochastic_rules.end())
            {
                //using insert, vs [] = because the [] requires they map_type to be default constructible
                stochastic_rules.insert({r.rname, r});
            }
            else
                std::cout << "stochastic rule name already exists in rule_set\n";
        }

        void addRule(SolvingRule<GraphType>& r)
        {
            if(deterministic_rules.find(r.rname) == deterministic_rules.end())
            {
                //using insert, vs [] = because the [] requires they map_type to be default constructible
                deterministic_rules.insert({r.rname, r});
            }
            else
                std::cout << "deterministic rule name already exists in rule_set\n";
        }

        void addRule(EveryRule<GraphType>& r)
        {
            if(every_rules.find(r.rname) == every_rules.end())
            {
                //using insert, vs [] = because the [] requires they map_type to be default constructible
                every_rules.insert({r.rname, r});
            }
            else
                std::cout << "every rule name already exists in rule_set\n";
        }

        void print()
        {
            std::cout << "Stochastic Rules: \n";
            for(auto& rule : stochastic_rules)
                std::cout << "\t" << rule.first << "\n";

            std::cout << "Deterministic Rules: \n";
            for(auto& rule : deterministic_rules)
                std::cout << "\t" << rule.first << "\n";

            std::cout << "Every Rules: \n";
            for(auto& rule : every_rules)
                std::cout << "\t" << rule.first << "\n";
        }
    };

} // DGGML
#endif //DGGML_GRAMMAR_H
