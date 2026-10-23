#ifndef DGGML_APPROXIMATESSA_HPP
#define DGGML_APPROXIMATESSA_HPP

//#include "PlantTypes.hpp"
//#include "PlantGrammar.hpp"

#include "ExpandedComplex2D.hpp"
#include "ExpandedComplex3D.hpp"

#include "YAGL_Graph.hpp"
#include "YAGL_Algorithms.hpp"
#include "YAGL_Graph_Parallel.hpp"

#include "Solver.h"

#include "ComponentMatchMap.hpp"
#include "IncrementalUpdate.hpp"
#include "IncrementalUpdate3D.hpp"

#include "AnalyzedGrammar.hpp"
#include "HelperFunctions.hpp"
#include "RandomFunctions.hpp"
#include "HelperStructs.hpp"

#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <functional>

#include "LivelinessMonitor.hpp"
#include <variant>
#include <omp.h>

                // Go fetch the node and edge changes and what it actually means
                // changes.print();
                // exit(0);
                // TODO: incrementally update the in memory set of matches
                //Once we have updated the graph, we need to invalidated any old matches and then search for the
                //new ones!

                //When it comes to match invalidation, removing a node, changing node type, or removing and edge
                //are the actions that lead to invalidation. Why does removal not trigger a re-search? Because our
                // pattern matcher would've already found any other sub pattern if it existed. Changing type, on the
                // other hand, does trigger a research. Adding new a node or edge only opens up a new search path and also does.
                // There are different apporaches to handle finding newly created matches. A potentially memory intensive
                // method would be to remember all completed and partial search paths. My method is a bit more brute force,
                // but only locally.
                // hierarchy: (geocells) -> rule instances -> components instances -> nodes
                // slow, but easy case to code - assume we know nothing and must search everywhere

                // should invalidate components and rule instances containing invalid components
    //            using graph_t = typename std::remove_reference<decltype(system_graph)>::type;
    //            auto removals = perform_invalidations<graph_t>(changes, component_matches,
    //                                                           grammar_analysis, rule_matches,
    //                                                           rule_map[k], cell_list);
    //            removals.print();

                // TODO: Get liveliness properties for finding matches?


namespace DGGML
{

    // ============================================================
    // ORIGINAL propensity evaluation (kept as reference).
    // Builds an induced subgraph copy per rule, then evaluates
    // the propensity lambda on the copy.
    // ============================================================
    template <typename RuleMatchesType, typename ComponentMatchesType,
              typename SystemGraphType, typename GrammarAnalysisType,
              typename MapType>
    inline double evaluate_propensity_with_induced_graph(
        RuleMatchesType& rule_matches, std::size_t r,
        ComponentMatchesType& component_matches,
        SystemGraphType& system_graph,
        GrammarAnalysisType& grammar_analysis,
        MapType& lhs_vertex_map)
    {
        auto induced_graph = induce_from_set(rule_matches[r], component_matches, system_graph);
        bool valid = construct_grammar_match_map(rule_matches[r], grammar_analysis, lhs_vertex_map, component_matches);
        if (!valid) return 0.0;
        return grammar_analysis.with_rules.at(rule_matches[r].name).propensity(induced_graph, lhs_vertex_map);
    }

    template <int Dim=3, typename T1, typename T2, typename T3, typename T4, typename M1, typename GraphType, typename GenType>
    void approximate_ssa(ComponentMatchMap<T1>& component_matches, AnalyzedGrammar<T2>& grammar_analysis, T3& rule_map,
                        T4& rule_matches, M1& model, std::size_t k, GeocellProperties<T1>& geocell_properties,
                        CellList<T2>& cell_list, LivelinessMonitor* liveliness_monitor,
                        GraphType& local_graph, GenType& local_gen,
                        // Per-cell read-only snapshots for cache-friendly propensity reads.
                        // Threads access only their own small private copies during the hot
                        // propensity loop; writes still go to the shared global maps via
                        // omp critical(structural_change).
                        std::unordered_map<std::size_t, typename T4::data_type>& local_rm,
                        ComponentMatchMap<T1>& local_cm)
    {
        auto& geoplex2D = model->geoplex2D;
        auto& system_graph = local_graph;
        auto& settings = model->settings;
        auto& gen = local_gen;
        auto reaction_radius = model->settings.MAXIMAL_REACTION_RADIUS;

        double delta_t, geocell_propensity;
        std::size_t events, steps;

        delta_t = 0.0; events = 0; steps = 0, geocell_propensity = 0.0;
        double tau = geocell_properties.tau;
        double exp_sample = geocell_properties.exp_sample;

        // TIMER instrumentation (always on)
        double timer_solver_init_ms = 0.0;
        double timer_propensity_ms = 0.0;
        double timer_prop_induce_ms = 0.0;
        double timer_prop_matchmap_ms = 0.0;
        double timer_prop_eval_ms = 0.0;
        double timer_prop_sum_ms = 0.0;
        double timer_ode_step_ms = 0.0;
        double timer_copy_back_ms = 0.0;
        double timer_event_select_ms = 0.0;
        double timer_rewrite_ms = 0.0;
        double timer_rw_vertex_map_ms = 0.0;
        double timer_rw_rule_apply_ms = 0.0;
        double timer_rw_rule_update_ms = 0.0;
        double timer_rw_invalidation_ms = 0.0;
        double timer_rw_graph_commit_ms = 0.0;
        double timer_find_matches_ms = 0.0;
        double timer_reinit_ms = 0.0;
        double timer_rebuild_snapshots_ms = 0.0;
        double timer_solving_rules_rebuild_ms = 0.0;
        double timer_other_ms = 0.0;
        std::size_t timer_prop_nrules = 0;
        std::size_t timer_prop_stale = 0;
        std::size_t ssa_inner_steps = 0;
        std::size_t stochastic_events = 0;
        auto timer_geocell_start = std::chrono::high_resolution_clock::now();

        // Reset liveliness monitor
        if (liveliness_monitor) {
            liveliness_monitor->ssa_steps = 0;
        }

        if(exp_sample <= 0.0)
        {
            double uniform_sample = RandomRealsBetween(0.0, 1.0)();
            // Warped waiting time
            exp_sample = -log(1-uniform_sample);
        }

        // ============================================================
        // EveryRule (clock-scheduled bulk-Jacobi rules) are now fired from
        // INSIDE the Gillespie loop below via a time-event queue, so a rule
        // whose period P = n_r * T_clock is SMALLER than the window DELTA fires
        // at its exact sub-window times, correctly interleaved with the ODE flow
        // and stochastic events -- rather than being batched at the window edge.
        // See the "Sub-DELTA EveryRule scheduling" block just before the loop and
        // branch (A) at the top of the loop for the realization.
        // ============================================================

        std::vector<std::size_t> solving_rules;

        for(auto& item : rule_map.at(k)) {
            // Use per-cell snapshot: avoids touching the shared global hash table
            if (auto it = local_rm.find(item); it != local_rm.end()) {
                if(it->second.category == "deterministic")
                    solving_rules.push_back(item);
            }
        }

        // the number of equations is defined by the parameters of the nodes
        // involved in all the ode rule types, it's up to the user to get the number per
        // rule right
        auto num_eq = 0;
        for(auto& item : solving_rules)
        {
            auto& name = local_rm.at(item).name;
            num_eq += grammar_analysis.solving_rules.find(name)->second.num_eq;
        }
        num_eq++; //one extra equation for tau

        // initialize the ODE solver
        auto _t_solver_init = std::chrono::high_resolution_clock::now();
        Solver ode_system(model, grammar_analysis, solving_rules,
                  rule_matches, local_rm,
                  local_cm,
                  k, geocell_propensity, tau, exp_sample, num_eq);
        timer_solver_init_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t_solver_init).count();

        // ============================================================
        // Sub-DELTA EveryRule scheduling (time-event queue).
        //
        // Grammar picture:
        //   {clock(tau), tau_a(i)(x_i)}_*  ->  {clock(tau), tau_a(j)(y_j)}_*
        //     with  Pulse(tau mod n_r T_clock - 2 p_r / Lambda ; 1 / Lambda)
        //            * prod_j delta( y_j - x_j - V_{r,j}([x_i | ...]) )
        //   plus the shared clock rule {clock(tau)}->{clock(tau)} solving dtau/dt = 1.
        //
        // Realization (option B, Lambda -> infinity): the shared clock tau is the
        //   PHYSICAL time. clock_base is its value ENTERING this window; local solver
        //   time t runs 0 -> DELTA, so absolute time = clock_base + t. Each EveryRule r
        //   has period P = n_r * T_clock (T_clock <= 0 falls back to settings.DELTA) and
        //   fires on the schedule phi_r + m P (m = 0,1,2,...), phi_r = p_r * P the phase
        //   offset. As Lambda -> infinity the Pulse collapses to an exact once-per-period
        //   gate. Instead of batching all boundaries at the window edge, we POST the next
        //   scheduled fire time of every rule to a min-heap and process them AS EVENTS
        //   interleaved with the ODE flow: this lets P < DELTA fire at its true sub-window
        //   times. geocell_properties.every_next_fire persists each rule's next absolute
        //   fire time across windows (shared global clock), so a rule may fire several
        //   times per window or once every few windows.
        // ============================================================
        const double clock_base = geocell_properties.clock_tau;
        const double window_end = clock_base + settings.DELTA;

        struct EveryEvent {
            double time;            // absolute physical fire time on the shared clock
            std::string name;       // EveryRule name
            bool operator>(const EveryEvent& o) const { return time > o.time; }
        };
        std::priority_queue<EveryEvent, std::vector<EveryEvent>,
                            std::greater<EveryEvent>> every_queue;

        // Period of an EveryRule by name (P = n_r * T_clock; T_clock<=0 -> DELTA).
        // nm is the EveryRule name, not the rule instance id.
        auto every_period = [&](const std::string& nm) -> double {

            auto& er = grammar_analysis.every_rules.at(nm);
            const double Tc = (er.T_clock > 0.0) ? er.T_clock : settings.DELTA;

            return static_cast<double>(er.n_r) * Tc;
        };

        // Seed the queue: one pending entry per EveryRule present in this geocell,
        // at its next scheduled absolute fire time (initialized to the phase offset,
        // then persisted/advanced across windows via every_next_fire).
        if(!grammar_analysis.every_rules.empty())
        {
            std::set<std::string> names_here;
            for(auto& item : rule_map.at(k))
            {
                auto it = local_rm.find(item);
                if(it != local_rm.end() && it->second.category == "every")
                    names_here.insert(it->second.name);
            }
            for(auto& nm : names_here)
            {

                auto& er = grammar_analysis.every_rules.at(nm);
                if(er.n_r <= 0) continue;

                // Compute the period of this EveryRule (P = n_r * T_clock; T_clock<=0 -> DELTA).
                // The period represents the time between successive firings of the rule,
                // and is used to schedule the next fire time.
                const double P = every_period(nm);

                if (P <= 0.0) continue;

                // The phase offset represents the time
                // within the period at which the rule first fires.
                const double phi = er.p_r * P;   // phase offset

                auto [nf_it, ins] = geocell_properties.every_next_fire.try_emplace(nm, phi);

                // A rule that first appears mid-simulation (e.g. a rule whose
                // pattern only exists once refinement creates it) or whose stored
                // next-fire lies in the past must NOT replay the missed fires as a
                // zero-time burst; skip forward to the first grid point
                // phi + m P at or after the start of this window.
                if(nf_it->second < clock_base - 1e-9 * P)
                {
                    const double skip = std::ceil((clock_base - nf_it->second) / P - 1e-9);
                    nf_it->second += std::max(0.0, skip) * P;
                }

                every_queue.push({nf_it->second, nm});

            }
        }

        // Fire one EveryRule as a bulk {..}_* simultaneous (Jacobi) rewrite:
        //   Phase 1: induce ALL matches from the CURRENT graph (read-only snapshots).
        //   Phase 2: apply the user update to each snapshot copy.
        //   Phase 3: commit only the principal (smallest-id LHS vertex = stencil centre)
        //            node -- ADDITIVELY (live += after - before, i.e. y_j := x_j + V) via
        //            the combine hook when supplied, else overwrite -- so overlapping
        //            stencils never conflict and every update reads inputs at old values.
        // The match set is regrouped at fire time so it stays correct even if a
        // stochastic event mutated the matches earlier in the window.
        // All rules passed in are due at the SAME physical time and fire as ONE
        // simultaneous (Jacobi) batch: every rule induces its matches and computes
        // its update from the same pre-fire snapshot, then all principal-node
        // commits are applied together (additive combine). Coincident rules with
        // different periods (e.g. coarse Fluid and sub-cycled Shock) therefore
        // never read each other's same-instant writes.
        auto fire_every_rules = [&](const std::vector<std::string>& enames)
        {
            struct EveryWork {
                std::string rule;                       // EveryRule name
                GraphType lhs;                          // read-only pre-fire snapshot
                GraphType rhs;                          // written copy
                std::map<std::size_t, std::size_t> m;   // pattern vertex -> system key
                std::size_t principal_key;              // system key of committed node
            };
            std::vector<EveryWork> work;
            std::set<std::string> seen_embeddings;

            // Phase 1: induce all matches of all due rules from the CURRENT graph
            {
                auto _rg = YAGL::make_graph_read_guard(system_graph);
                for(const auto& ename : enames)
                {
                    for(auto id : rule_map.at(k))
                    {
                        auto it = local_rm.find(id);
                        if(it == local_rm.end()) continue;
                        auto& inst = it->second;
                        if(inst.category != "every" || inst.name != ename) continue;
                        EveryWork w;
                        w.rule = ename;
                        construct_grammar_match_map(inst, grammar_analysis, w.m, local_cm);
                        if(w.m.empty()) continue;
                        // A rewrite that PRESERVES a node can leave an older match
                        // alive next to a freshly rediscovered identical one; the
                        // same embedding must only contribute once per fire.
                        {
                            std::string key = ename;
                            for(auto& kv : w.m)
                                key += ',' + std::to_string(kv.first) + ':' + std::to_string(kv.second);
                            if(!seen_embeddings.insert(std::move(key)).second) continue;
                        }
                        w.lhs = induce_from_set(inst, local_cm, system_graph);
                        w.rhs = w.lhs;
                        w.principal_key = w.m.begin()->second;   // stencil centre
                        work.push_back(std::move(w));
                    }
                }
            }

            // Phase 2: apply the update to each snapshot copy
            for(auto& w : work)
                grammar_analysis.every_rules.at(w.rule).update(w.lhs, w.rhs, w.m, w.m);

            // Phase 3: commit principal-node data simultaneously
            {
                auto _wg = YAGL::make_graph_write_guard(system_graph);
                for(auto& w : work)
                {
                    auto& erule = grammar_analysis.every_rules.at(w.rule);
                    auto dst = system_graph.findNode(w.principal_key);
                    auto src = w.rhs.findNode(w.principal_key);
                    auto pre = w.lhs.findNode(w.principal_key);
                    if(dst != system_graph.node_list_end()
                       && src != w.rhs.node_list_end()
                       && pre != w.lhs.node_list_end())
                    {
                        if(erule.combine)
                            erule.combine(dst->second.getData(),
                                          pre->second.getData(),
                                          src->second.getData());
                        else
                            dst->second = src->second;
                    }
                }
            }

            // The bulk write changed node data; refresh the continuous ODE state to
            // follow the graph WITHOUT disturbing the Gillespie clock (t, tau). No-op
            // for pure clock models (only the tau equation present).
            ode_system.resync_state();
        };

        while(delta_t < settings.DELTA)
        {

            // (A) Fire any EveryRule events whose scheduled time has arrived
            //     (<= current absolute time). All rules due at the same physical
            //     instant (within a small tolerance) fire together as ONE
            //     simultaneous (Jacobi) batch; distinct instants fire in time order.
            //     Each fired rule is rescheduled one period ahead and re-posted
            //     (this window if it still lands inside it, else persisted for a
            //     later window).
            while(!every_queue.empty()
                  && every_queue.top().time <= clock_base + delta_t + 1e-12)
            {
                const double t_batch = every_queue.top().time;
                const double tie_tol = 1e-9 * settings.DELTA;
                std::vector<EveryEvent> batch;
                while(!every_queue.empty()
                      && every_queue.top().time <= t_batch + tie_tol)
                {
                    batch.push_back(every_queue.top());
                    every_queue.pop();
                }

                std::vector<std::string> due;
                due.reserve(batch.size());
                for(auto& ev : batch) due.push_back(ev.name);
                fire_every_rules(due);

                for(auto& ev : batch)
                {
                    const double P = every_period(ev.name);
                    const double nxt = ev.time + P;
                    geocell_properties.every_next_fire[ev.name] = nxt;

                    if(P > 0.0 && nxt < window_end - 1e-12)
                        every_queue.push({nxt, ev.name});
                }
            }

            // the propensities calculated for a particular rule
            std::vector<std::pair<std::size_t, double>> propensity_space;

            if(tau < exp_sample)
            {
                auto delta_t_before_step = delta_t;  // track for stuck detection
                auto _t_prop_start = std::chrono::high_resolution_clock::now();
                // STEP(1) : sum all of the rule propensities
                for(const auto& r : rule_map.at(k))
                {
                    std::vector<std::size_t> inducers;
                    // Use per-cell snapshot: reads stay in this thread's private cache lines
                    auto lrm_it = local_rm.find(r);
                    if (lrm_it == local_rm.end()) continue;  // stale entry, skip
                    auto& inst = lrm_it->second;
                    if(inst.category != "stochastic")
                        continue;

                    timer_prop_nrules++;

                    auto _t_ind = std::chrono::high_resolution_clock::now();

                    // auto induced_graph = induce_from_set(inst, local_cm, system_graph);
                    // after:
                    auto induced_graph = [&]() {
                        auto _rg = YAGL::make_graph_read_guard(system_graph);
                        return induce_from_set(inst, local_cm, system_graph);
                    }();

                    timer_prop_induce_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_ind).count();

                    std::map<std::size_t, std::size_t> lhs_vertex_map;
                    auto _t_mm = std::chrono::high_resolution_clock::now();
                    construct_grammar_match_map(inst, grammar_analysis, lhs_vertex_map, local_cm);
                    timer_prop_matchmap_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_mm).count();

                    auto _t_ev = std::chrono::high_resolution_clock::now();

                    auto rho = grammar_analysis.with_rules.at(inst.name).propensity(
                        induced_graph, lhs_vertex_map
                    );

                    timer_prop_eval_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_ev).count();

                    if (rho == 0.0) timer_prop_stale++;
                    propensity_space.push_back({r,rho});
                }

                auto _t_sum_s = std::chrono::high_resolution_clock::now();
                geocell_propensity = 0.0;
                for(auto& [name, data] : grammar_analysis.with_rules)
                {
                    auto sum = 0.0;
                    for(auto& item : propensity_space)
                    {
                        auto it = local_rm.find(item.first);
                        if (it != local_rm.end() && it->second.name == name) sum += item.second;
                    }
                    if (sum > 0.0) geocell_propensity += sum;
                }
                timer_prop_sum_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - _t_sum_s).count();
                timer_propensity_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - _t_prop_start).count();

                // STEP(2) : solve the system of ODES
                auto _t_ode_start = std::chrono::high_resolution_clock::now();
                // Cap the ODE output at the next scheduled EveryRule boundary (in
                // solver-local time) so the integrator stops exactly there and the
                // clock event fires on time. With no pending boundary this is just
                // the window end (t_final), i.e. the original behaviour.
                double _every_cap = settings.DELTA;   // solver-local time of window end
                if(!every_queue.empty())
                    _every_cap = std::min(_every_cap, every_queue.top().time - clock_base);

                ode_system.step(_every_cap);

                timer_ode_step_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - _t_ode_start).count();

                {
                    auto _t_cb_start = std::chrono::high_resolution_clock::now();
                    ode_system.copy_back();
                    timer_copy_back_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_cb_start).count();
                }

                // STEP(3) : advance the loop timer
                delta_t = ode_system.t;

                // TODO: Allow this to be optional
                if (liveliness_monitor) {
                    liveliness_monitor->notifyTimeAdvanced(delta_t);
                    liveliness_monitor->notifyPropensityCalculated(k, geocell_propensity);
                }

                steps++;
                ssa_inner_steps++;
                // Alternate STEP(2) : use forward euler to solve the TAU ODE
    //            tau += geocell_propensity*settings.DELTA_T_MIN;
    //            std::cout << "Current tau: " << tau << "\n";
                //Alternate STEP(3)
                //delta_t += settings.DELTA_T_MIN;
            }

            // if we get over our threshold an event can occur
            if(tau >= exp_sample) {

                // --- Event selection ---
                auto _t_esel = std::chrono::high_resolution_clock::now();
                auto lam = [](double& s, auto& rp) { return s+rp.second; };
                auto sum = std::accumulate(propensity_space.begin(), propensity_space.end(), 0.0, lam);

                auto local_sample = RandomRealsBetween(0.0, 1.0)();
                auto eventFired = 0;
                auto local_progress = local_sample*geocell_propensity - propensity_space[0].second;
                while(local_progress > 0.0)
                {
                    eventFired++;
                    local_progress -= propensity_space[eventFired].second;
                }
                timer_event_select_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - _t_esel).count();

                auto fired_id = propensity_space[eventFired].first;
                // Use local snapshot for fired rule lookup (already in our private cache)
                auto inst = local_rm.count(fired_id) ? local_rm.at(fired_id) : rule_matches.at(fired_id);
                auto fired_name = inst.name;

                std::pair<DGGML::RewriteUpdates, DGGML::Invalidations> results;

                auto do_rewrite = [&]() {
                    liveliness_monitor->notifyRuleFired(fired_name);

                    auto _t_rewrite_start = std::chrono::high_resolution_clock::now();
                    if (Dim  == 2) {
                        results = perform_invalidations_and_rewrite(inst, component_matches, gen, grammar_analysis,
                                                                                     system_graph, rule_matches, rule_map.at(k),
                                                                                     cell_list, geocell_properties);
                    } else {
                        results = DGGML3D::perform_invalidations_and_rewrite(inst, component_matches, gen, grammar_analysis,
                                                 system_graph, rule_matches, rule_map, rule_map.at(k),
                                                 cell_list, geocell_properties);
                    }
                    timer_rewrite_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_rewrite_start).count();
                    timer_rw_vertex_map_ms  += DGGML3D::g_rewrite_timings.vertex_map_ms;
                    timer_rw_rule_apply_ms  += DGGML3D::g_rewrite_timings.rule_apply_ms;
                    timer_rw_rule_update_ms += DGGML3D::g_rewrite_timings.rule_update_ms;
                    timer_rw_invalidation_ms += DGGML3D::g_rewrite_timings.invalidation_ms;
                    timer_rw_graph_commit_ms += DGGML3D::g_rewrite_timings.graph_commit_ms;

                    auto [changes, removals] = results;
                    liveliness_monitor->notifyComponentsInvalidatedValidated(changes, removals);

                    auto _t_match_start = std::chrono::high_resolution_clock::now();
                    if (Dim == 2) {
                        find_new_matches(changes, system_graph, component_matches,grammar_analysis,
                                         rule_matches, rule_map.at(k), cell_list,
                                         geoplex2D,k, reaction_radius, geocell_properties);
                    } else {
                        // This buffers the changes
                        // and if a type moves out of the geo cell it removes it from the current loop
                        // and buffers it to be passed on to the next geocell during the sync.
                        DGGML3D::find_new_matches(changes, system_graph, component_matches,grammar_analysis,
                                         rule_matches, rule_map.at(k), cell_list,
                                         model->geoplex2D,k, reaction_radius, geocell_properties);
                    }
                    timer_find_matches_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_match_start).count();

                    geocell_properties.current_rules_fired++;
                    geocell_properties.total_rules_fired++;
                    // Rebuild solving_rules after event: rule_map.at(k) and
                    // rule_matches were just updated by find_new_matches above.
                    {
                        auto _t_sr = std::chrono::high_resolution_clock::now();
                        solving_rules.clear();
                        for(auto& item : rule_map.at(k))
                            if(rule_matches.at(item).category == "deterministic")
                                solving_rules.push_back(item);
                        timer_solving_rules_rebuild_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - _t_sr).count();
                    }
                };

                // Helper lambda: rebuild local propensity-loop snapshots from
                // whatever component_matches / rule_matches was passed to this call
                // (may be global or per-cell depending on dimension).
                /*
                auto rebuild_snapshots = [&]() {
                    local_rm.clear();
                    local_cm.matches.clear();
                    for (auto rid : rule_map.at(k)) {
                        if (auto it = rule_matches.matches.find(rid); it != rule_matches.matches.end()) {
                            local_rm[rid] = it->second;
                            for (auto cid : it->second.components) {
                                if (auto ci = component_matches.matches.find(cid); ci != component_matches.matches.end())
                                    local_cm.matches[cid] = ci->second;
                            }
                        }
                    }
                };
                */

                auto rebuild_snapshots = [&]() {
                local_rm.clear();
                local_cm.matches.clear();
                for (auto rid : rule_map.at(k)) {
                    auto it = rule_matches.matches.find(rid);
                    if (it == rule_matches.matches.end()) continue;
                    // Skip stale rules whose components were invalidated in another geocell
                    bool valid = true;
                    for (auto cid : it->second.components) {
                        if (component_matches.matches.find(cid) == component_matches.matches.end()) {
                            valid = false; break;
                        }
                    }
                    if (!valid) continue;
                    local_rm[rid] = it->second;
                    for (auto cid : it->second.components) {
                        if (auto ci = component_matches.matches.find(cid); ci != component_matches.matches.end())
                            local_cm.matches[cid] = ci->second;
                    }
                }
            };

                do_rewrite();

                // Worklist approach: approximate_ssa operates on per-cell
                // component/rule maps supplied by the caller (SimulationRunnerParallel
                // passes component_matches_by_cell[k] / rule_matches_by_cell[k]).
                // Rewrites go to those isolated copies; global maps are updated
                // after the parallel phase via conflict detection + merge_from_per_cell.
                // No critical section needed here.
                // do_rewrite();
                {
                    auto _t_rs = std::chrono::high_resolution_clock::now();
                    rebuild_snapshots();
                    timer_rebuild_snapshots_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_rs).count();
                }

                tau = 0.0;
                double uniform_sample = RandomRealsBetween(0.0, 1.0)();
                exp_sample = -log(1-uniform_sample);
                ode_system.user_data.waiting_time = exp_sample;
                stochastic_events++;
                {
                    auto _t_reinit_start = std::chrono::high_resolution_clock::now();
                    ode_system.reinit(solving_rules, local_rm);
                    timer_reinit_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - _t_reinit_start).count();
                }
            }

            //break; //makes only one reaction occur break point
            liveliness_monitor->notifySSAstep();
        }

        if (ssa_inner_steps > 0) {
            auto timer_geocell_end = std::chrono::high_resolution_clock::now();
            double timer_total_ms = std::chrono::duration<double, std::milli>(timer_geocell_end - timer_geocell_start).count();
            long int nst = 0, nfe_final = 0;
            ERKStepGetNumSteps(ode_system.arkode_mem, &nst);
            nst += ode_system.global_nstep;
            ERKStepGetNumRhsEvals(ode_system.arkode_mem, &nfe_final);
            std::size_t total_nfe = ode_system.global_nfe + nfe_final;
            double accounted = timer_solver_init_ms + timer_propensity_ms + timer_ode_step_ms
                             + timer_copy_back_ms + timer_event_select_ms + timer_rewrite_ms
                             + timer_find_matches_ms + timer_reinit_ms
                             + timer_rebuild_snapshots_ms + timer_solving_rules_rebuild_ms;
            timer_other_ms = timer_total_ms - accounted;

            std::cout << std::fixed << std::setprecision(3)
                      << "  [geocell " << k << "] total=" << timer_total_ms << "ms"
                      << "  steps=" << ssa_inner_steps << "  events=" << stochastic_events
                      << "  ode_steps=" << nst << "  rhs_evals=" << total_nfe << "\n"
                      << "    solver_init=" << timer_solver_init_ms << "ms"
                      << "  propensity=" << timer_propensity_ms << "ms"
                      << " (induce=" << timer_prop_induce_ms
                      << " matchmap=" << timer_prop_matchmap_ms
                      << " eval=" << timer_prop_eval_ms
                      << " sum=" << timer_prop_sum_ms
                      << " n=" << timer_prop_nrules
                      << " stale=" << timer_prop_stale << ")"
                      << "\n    ode_solve=" << timer_ode_step_ms << "ms"
                      << "  copy_back=" << timer_copy_back_ms << "ms"
                      << "  event_select=" << timer_event_select_ms << "ms"
                      << "  rewrite=" << timer_rewrite_ms << "ms"
                      << " (vertex_map=" << timer_rw_vertex_map_ms
                      << " rule_apply=" << timer_rw_rule_apply_ms
                      << " rule_update=" << timer_rw_rule_update_ms
                      << " invalidation=" << timer_rw_invalidation_ms
                      << " graph_commit=" << timer_rw_graph_commit_ms << ")"
                      << "  find_match=" << timer_find_matches_ms << "ms"
                      << "  reinit=" << timer_reinit_ms << "ms"
                      << "  rebuild_snapshots=" << timer_rebuild_snapshots_ms << "ms"
                      << "  solving_rules_rebuild=" << timer_solving_rules_rebuild_ms << "ms"
                      << "  OTHER=" << timer_other_ms << "ms"
                      << std::endl;
        }

        // Advance the shared clock by one synchronization window (dtau/dt = 1).
        //
        // NOTE: this clock tau is the PHYSICAL time of the shared clock rule
        //   {clock(tau)} -> {clock(tau)} solving dtau/dt = 1, i.e. it tracks the
        //   ODE solver's INDEPENDENT variable ode_system.t (which runs 0 -> DELTA
        //   each call), NOT the solver's warped state variable `tau` above (whose
        //   equation is dtau/dt = total propensity, the Gillespie waiting clock).
        //
        //   We add settings.DELTA (the uniform window quantum) rather than the
        //   local delta_t = ode_system.t. delta_t overshoots DELTA by up to one
        //   DELTA_T_MIN step and by a DIFFERENT amount in every geocell, so
        //   accumulating it would let each geocell's copy of the SHARED clock drift
        //   independently and diverge from the engine's global time grid
        //   (windows * DELTA). Adding DELTA keeps every geocell's clock_tau exactly
        //   equal to the global elapsed physical time, which is what a shared clock
        //   requires for consistent cross-geocell EveryRule firing.
        geocell_properties.clock_tau += settings.DELTA;

        geocell_properties.tau = tau;
        geocell_properties.exp_sample = exp_sample;

    }

    // Backward-compatible overload (serial runners): builds temporary per-call snapshots
    // and delegates to the snapshot-aware version above.
    template <int Dim=3, typename T1, typename T2, typename T3, typename T4, typename M1, typename GraphType, typename GenType>
    void approximate_ssa(ComponentMatchMap<T1>& component_matches, AnalyzedGrammar<T2>& grammar_analysis, T3& rule_map,
                        T4& rule_matches, M1& model, std::size_t k, GeocellProperties<T1>& geocell_properties,
                        CellList<T2>& cell_list, LivelinessMonitor* liveliness_monitor,
                        GraphType& local_graph, GenType& local_gen)
    {
        std::unordered_map<std::size_t, typename T4::data_type> local_rm;
        ComponentMatchMap<T1> local_cm;
        if (auto it = rule_map.find(k); it != rule_map.end()) {
            for (auto rid : it->second) {
                if (auto rit = rule_matches.matches.find(rid); rit != rule_matches.matches.end()) {
                    local_rm[rid] = rit->second;
                    for (auto cid : rit->second.components) {
                        if (auto ci = component_matches.matches.find(cid); ci != component_matches.matches.end())
                            local_cm.matches[cid] = ci->second;
                    }
                }
            }
        }

        approximate_ssa<Dim>(component_matches, grammar_analysis, rule_map, rule_matches,
                             model, k, geocell_properties, cell_list, liveliness_monitor,
                             local_graph, local_gen, local_rm, local_cm);
    }

}

#endif //DGGML_APPROXIMATESSA_HPP
