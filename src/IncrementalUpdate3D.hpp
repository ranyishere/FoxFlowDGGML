#ifndef DGGML_INCREMENTALUPDATE_3D_HPP
#define DGGML_INCREMENTALUPDATE_3D_HPP

#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <map>
#include <set>

#include "ComponentMatchMap.hpp"
#include "YAGL_Algorithms.hpp"
#include "YAGL_Graph.hpp"
#include "YAGL_Graph_Parallel.hpp"
#include "AnalyzedGrammar.hpp"
#include "CellList.hpp" 
#include "pattern_matching.hpp"
#include "pattern_matching_3d.hpp"
#include "phi_functions.hpp"
#include "HelperStructs.hpp"
#include "DGGML.h"

#include "IncrementalUpdate.hpp" // For common structures
#include <chrono>
#include <iomanip>

namespace DGGML3D
{

    // Accumulated timings from the last perform_invalidations_and_rewrite call.
    // Single-threaded — safe to read immediately after each call.
    struct RewriteTimings {
        double vertex_map_ms   = 0.0; // building lhs_vertex_map + induced subgraphs
        double rule_apply_ms   = 0.0; // node/edge destroy/create + type updates
        double rule_update_ms  = 0.0; // grammar_analysis.with_rules.update()
        double invalidation_ms = 0.0; // perform_invalidations (stale rule cleanup)
        double graph_commit_ms = 0.0; // writing changes back to system_graph
        double total_ms        = 0.0;

        void reset() { *this = RewriteTimings{}; }

        void print() const {
            std::cout << std::fixed << std::setprecision(4)
                      << "      rewrite breakdown:"
                      << "  vertex_map=" << vertex_map_ms << "ms"
                      << "  rule_apply=" << rule_apply_ms << "ms"
                      << "  rule_update=" << rule_update_ms << "ms"
                      << "  invalidation=" << invalidation_ms << "ms"
                      << "  graph_commit=" << graph_commit_ms << "ms"
                      << "  (other=" << total_ms - vertex_map_ms - rule_apply_ms
                                                 - rule_update_ms - invalidation_ms
                                                 - graph_commit_ms << "ms)"
                      << std::endl;
        }
    };
    inline thread_local RewriteTimings g_rewrite_timings;

    template<typename GraphType>
    DGGML::Invalidations perform_invalidationsOld(DGGML::RewriteUpdates& changes,
                                        DGGML::ComponentMatchMap<std::size_t>& component_matches,
                                        DGGML::AnalyzedGrammar<GraphType>& grammar_analysis,
                                        DGGML::RuleMatchMap<std::size_t>& rule_matches,
                                        std::map<std::size_t, std::vector<std::size_t>>& global_rule_map,
                                        DGGML::CellList<GraphType>& cell_list, DGGML::GeocellProperties<std::size_t>& geocell_properties)
    {
        DGGML::Invalidations removals;
        auto& component_invalidations = removals.component_invalidations;

        for(auto& [k1, m] : component_matches)
        {
            for(auto i = 0; i < m.match.size(); i++)
            {
                auto k2 = m.match[i];
                if(changes.node_removals.find(k2) != changes.node_removals.end())
                {
                    component_invalidations.insert(k1);
                }
                for(auto j = 0; j < m.match.size(); j++)
                {
                    auto k3 = m.match[j];
                    auto& c = grammar_analysis.unique_components[m.type];
                    std::vector<std::size_t> ordering = grammar_analysis.orderings[m.type];
                    if(changes.edge_removals.find({k2, k3}) != changes.edge_removals.end()
                       && c.adjacent(ordering[i], ordering[j]))
                    {
                        component_invalidations.insert(k1);
                    }
                }
            }
        }

        for(auto& item : component_invalidations)
        {
            geocell_properties.invalidated_components.insert(item);
        }

        for(auto& item : component_invalidations)
        {
            auto iter = component_matches.find(item);
            cell_list.erase(iter);
            component_matches.erase(item);
        }

        auto& rule_invalidations = removals.rule_invalidations;
        // Clean stale rules from ALL geocells in the global map
        for(auto& [gcell_key, local_rule_map] : global_rule_map)
        {
            for(auto& k : local_rule_map)
            {
                auto it = rule_matches.find(k);
                if(it == rule_matches.end()) continue;
                auto& inst = it->second;
                for(auto& c : inst.components)
                {
                    if(component_invalidations.find(c) != component_invalidations.end())
                    {
                        rule_invalidations.insert(k);
                    }
                }
            }
        }

        for(auto& item : rule_invalidations) {
            rule_matches.erase(item);
            // Remove from every geocell's local map that contains it
            for(auto& [gcell_key, local_rule_map] : global_rule_map)
            {
                auto pos = std::find(local_rule_map.begin(), local_rule_map.end(), item);
                if(pos != local_rule_map.end())
                    local_rule_map.erase(pos);
            }
        }

        return removals;
    }

        template<typename GraphType>
    DGGML::Invalidations perform_invalidations(DGGML::RewriteUpdates& changes,
                                        DGGML::ComponentMatchMap<std::size_t>& component_matches,
                                        DGGML::AnalyzedGrammar<GraphType>& grammar_analysis,
                                        DGGML::RuleMatchMap<std::size_t>& rule_matches,
                                        std::vector<std::size_t>& rule_map,
                                        DGGML::CellList<GraphType>& cell_list, DGGML::GeocellProperties<std::size_t>& geocell_properties)
    {
        DGGML::Invalidations removals;
        auto& component_invalidations = removals.component_invalidations;

        for(auto& [k1, m] : component_matches)
        {
            for(auto i = 0; i < m.match.size(); i++)
            {
                auto k2 = m.match[i];
                if(changes.node_removals.find(k2) != changes.node_removals.end())
                {
                    component_invalidations.insert(k1);
                }
                for(auto j = 0; j < m.match.size(); j++)
                {
                    auto k3 = m.match[j];
                    auto& c = grammar_analysis.unique_components[m.type];
                    std::vector<std::size_t> ordering = grammar_analysis.orderings[m.type];
                    if(changes.edge_removals.find({k2, k3}) != changes.edge_removals.end()
                       && c.adjacent(ordering[i], ordering[j]))
                    {
                        component_invalidations.insert(k1);
                    }
                }
            }
        }

        for(auto& item : component_invalidations)
        {
            geocell_properties.invalidated_components.insert(item);
        }

        for(auto& item : component_invalidations)
        {
            auto iter = component_matches.find(item);
            cell_list.erase(iter);
            component_matches.erase(item);
        }

        auto& rule_invalidations = removals.rule_invalidations;
        for(auto& k : rule_map)
        {
            auto& inst = rule_matches[k];
            for(auto& c : inst.components)
            {
                if(component_invalidations.find(c) != component_invalidations.end())
                {
                    rule_invalidations.insert(k);
                }
            }
        }

        for(auto& item : rule_invalidations) {
            rule_matches.erase(item);
            rule_map.erase(std::find(rule_map.begin(), rule_map.end(), item));
        }

        return removals;
    }

    template <typename GraphType, typename MatchType>
    std::pair<DGGML::RewriteUpdates, DGGML::Invalidations> perform_invalidations_and_rewrite(DGGML::RuleMatch<std::size_t>& inst,
                                                     MatchType& component_match_set,
                                                     DGGML::KeyGenerator<std::size_t>& gen,
                                                     DGGML::AnalyzedGrammar<GraphType>& grammar_analysis,
                                                     GraphType& system_graph,
                                                     DGGML::RuleMatchMap<std::size_t>& rule_matches,
                                                     std::map<std::size_t, std::vector<std::size_t>>& global_rule_map,
                                                     std::vector<std::size_t>& local_rule_map,
                                                     DGGML::CellList<GraphType>& cell_list,
                                                     DGGML::GeocellProperties<std::size_t>& geocell_properties)
    {
        g_rewrite_timings.reset();
        auto _t_total = std::chrono::high_resolution_clock::now();

        DGGML::RewriteUpdates changes;
        std::string rname = inst.name;

        // --- Phase 1: build vertex maps and induced subgraphs ---
        auto _t0 = std::chrono::high_resolution_clock::now();
        std::map<std::size_t, std::size_t> lhs_vertex_map;
        std::vector<std::size_t> left, mid, right;
        for(auto& m : grammar_analysis.ccuv_mappings[rname])
            for (auto &[k, v]: m)
            { left.push_back(k); mid.push_back(v); }

        for(auto& c : inst.components)
            for(auto& k : component_match_set[c].match)
                right.push_back(k);

        for(auto i = 0; i < left.size(); i++)
            lhs_vertex_map[left[i]] = right[i];

        auto& rewrite = grammar_analysis.with_rewrites.at(rname);
        auto lhs_match = [&]() {
            auto _rg = YAGL::make_graph_read_guard(system_graph);
            return YAGL::induced_subgraph(system_graph, right);
        }();
        auto lhs_match_copy = lhs_match;
        auto rhs_rule_copy = grammar_analysis.with_rules.at(rname).rhs_graph;
        auto rhs_vertex_map = lhs_vertex_map;
        g_rewrite_timings.vertex_map_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t0).count();

        // --- Phase 2: apply rule (node/edge destroy, create, type updates) ---
        _t0 = std::chrono::high_resolution_clock::now();
        for(auto& k : rewrite.node_set_destroy)
        {
            rhs_vertex_map.erase(k);
            lhs_match_copy.removeNode(lhs_match_copy.findNode(lhs_vertex_map[k])->second);
            changes.node_removals.insert(lhs_vertex_map[k]);
        }

        for(auto& k : rewrite.node_set_create)
        {
            auto n = rhs_rule_copy.findNode(k)->second;
            decltype(n) node(gen.get_key(), n.getData());
            lhs_match_copy.addNode(node);
            rhs_vertex_map[k] = node.getKey();
            changes.node_updates.insert(rhs_vertex_map[k]);
        }

        for(auto& [u, v] : rewrite.edge_set_destroy)
        {
            lhs_match_copy.removeEdge(lhs_vertex_map[u], lhs_vertex_map[v]);
            changes.edge_removals.insert({lhs_vertex_map[u], lhs_vertex_map[v]});
        }

        for(auto& [u, v] : rewrite.edge_set_create) {
            lhs_match_copy.addEdge(rhs_vertex_map[u], rhs_vertex_map[v]);
            changes.edge_updates.insert({rhs_vertex_map[u], rhs_vertex_map[v]});
        }

        for(auto& [k, _] : rhs_rule_copy.getNodeSetRef())
        {
            auto& v1 = lhs_match_copy[rhs_vertex_map[k]];
            auto& v2 = rhs_rule_copy[k];
            if(v1.type != v2.type) {
                v1.setData(v2.data);

                changes.node_removals.insert(rhs_vertex_map[k]);
                changes.node_updates.insert(rhs_vertex_map[k]);
            }

        }

        g_rewrite_timings.rule_apply_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t0).count();

        // --- Phase 3: grammar rule update (user callback) ---
        _t0 = std::chrono::high_resolution_clock::now();
        grammar_analysis.with_rules.at(rname).update(lhs_match, lhs_match_copy, lhs_vertex_map, rhs_vertex_map);
        g_rewrite_timings.rule_update_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t0).count();

	// For surviving nodes that actually moved: invalidate the stale component
        // match (prevents duplicates in component_matches) and re-seed the search
        // at the new position. Type-change case already handled above.
        for (auto& [k, _] : rhs_rule_copy.getNodeSetRef()) {
            if (rewrite.node_set_create.find(k) == rewrite.node_set_create.end()) {
                auto mapped = rhs_vertex_map.at(k);
                const auto& pd_before = lhs_match[mapped];
                const auto& pd_after  = lhs_match_copy[mapped];
                if (pd_before.position[0] != pd_after.position[0] ||
                    pd_before.position[1] != pd_after.position[1] ||
                    pd_before.position[2] != pd_after.position[2]) {
                    changes.node_removals.insert(mapped); // evict stale component
                    changes.node_updates.insert(mapped);  // re-seed search at new pos
                }
            }
        }

        // --- Phase 4: invalidate stale rules across all geocells ---
        _t0 = std::chrono::high_resolution_clock::now();
        auto removals = DGGML3D::perform_invalidations(changes, component_match_set, grammar_analysis,
                                              rule_matches, local_rule_map, cell_list, geocell_properties);
        g_rewrite_timings.invalidation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t0).count();

        // --- Phase 5: commit changes to system_graph ---
        _t0 = std::chrono::high_resolution_clock::now();
        {
            auto _wg = YAGL::make_graph_write_guard(system_graph);
            for(auto& k : rewrite.node_set_destroy)
                system_graph.removeNode(system_graph.findNode(lhs_vertex_map[k])->second);

            for(auto& [u, v] : rewrite.edge_set_destroy)
                system_graph.removeEdge(lhs_vertex_map[u], lhs_vertex_map[v]);

            for(auto& [k, v] : lhs_match_copy.getNodeSetRef())
            {
                auto res = system_graph.findNode(k);
                if(res != system_graph.node_list_end())
                    res->second = v;
                else
                    system_graph.addNode(v);
            }

            for(auto& [u, v] : rewrite.edge_set_create)
                system_graph.addEdge(rhs_vertex_map[u], rhs_vertex_map[v]);
        }
        g_rewrite_timings.graph_commit_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t0).count();

        g_rewrite_timings.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _t_total).count();

        return std::make_pair(changes, removals);
    }


    /**
     * Only finds new matches locally for 
     */
    template<typename GraphType, typename CellComplexType>
    void find_new_matches(DGGML::RewriteUpdates& changes,
                          GraphType& system_graph,
                          DGGML::ComponentMatchMap<std::size_t>& component_matches,
                          DGGML::AnalyzedGrammar<GraphType>& grammar_analysis,
                          DGGML::RuleMatchMap<std::size_t>& rule_matches,
                          std::vector<std::size_t>& rule_map,
                          DGGML::CellList<GraphType>& cell_list,
                          CellComplexType& geoplex, // Generalized name for 3D
                          std::size_t cell_k,
                          double reaction_radius,
                          DGGML::GeocellProperties<std::size_t>& geocell_properties)
    {
        // Used to hold candidate nodes for subgraph isomorphism search.
        // Initially seeded with directly updated nodes, then expanded to include their neighbors.
        std::set<std::size_t> candidate_nodes = changes.node_updates;
        // Grab each vertex from each edge that was updated.
        for(auto& edge : changes.edge_updates)
        {
            candidate_nodes.insert(edge.first);
            candidate_nodes.insert(edge.second);
        }

        std::set<std::size_t> inducers;

        // MAX_DEPTH
        const int DEPTH_SEARCH = 6;

        // Hold a shared read lock for the entire match-finding phase.
        // Multiple geocells can DFS concurrently; Phase-5 write guards
        // block until all readers have released this lock.
        auto _rg = YAGL::make_graph_read_guard(system_graph);

        // For each node, find the induced subgraph that contains only those vertices.
        // You find them using depth first search
        // FIXME: ? This should be dependent on the lhs
        for(auto n : candidate_nodes)
        {
            auto res = YAGL::recursive_dfs(system_graph, n, DEPTH_SEARCH);
            for(auto& item : res) {
                inducers.insert(item);
            }
        }

        // This is the induced subgraph from the vertices that we have.
        auto candidate_graph = YAGL::induced_subgraph(system_graph, inducers);

        // Q: How do I know validated components is right?
        // Loop through the patterns in our grammar and find the correct candidate graph.

        // validated_components is the list of components that
        // we know are valid based on the directly updated nodes and edges.
        std::vector<std::size_t> validated_components;

        // For each unique component in the grammar, check if it exists in the candidate graph.
        // then we can validate the component matches that correspond to that unique component.
        for(auto& [k, pattern] : grammar_analysis.unique_components) {
            auto matches = YAGL::subgraph_isomorphism2(pattern, candidate_graph);
            if (!matches.empty()) {

                // Print out the matches to check
                // for each match in matches, print out the mapping of pattern vertices to candidate graph vertices

                for (auto &m: matches) {
                    bool accepted = false;

                    // v2 is the vertex in the candidate graph,
                    // n is the updated node in the system graph.
                    // If they match,
                    // we know this component is accepted and can skip edge checks.
                    // We can also add it to the validated components list to be re-seeded for future matches.
                    // v1 is the vertex in the pattern graph, which we can use to look up the component type and ordering.
                    // for each match, we loop through the updated nodes and check if any of them correspond to a vertex in the match.
                    for (auto& [v1, v2] : m) {

                        // Handle node updates first:
                        // if the updated node is part of the match,
                        // we know it's accepted and can skip edge checks
                        for (auto& n: changes.node_updates) {

                            if (v2 == n) {
                                accepted = true;
                                std::vector<typename GraphType::key_type> order;
                                std::vector<typename GraphType::key_type> match;
                                for (auto &[key, value]: m) {
                                    order.push_back(key);
                                    match.push_back(value);
                                }

                                DGGML::ComponentMatch<typename GraphType::key_type> inst;
                                inst.match = match;
                                inst.order = order;
                                inst.type = k;
                                inst.anchor = match[0];

                                auto res = component_matches.insert(inst);

                                if (res.second)
                                    validated_components.push_back(res.first->first);
                                break;

                            }

                        }
                        // std::cout << "accepted: " << accepted << std::endl;
                        if (accepted) break;
                    }

                    if (accepted) continue;

                    for (auto& [v1, v2] : m) {
                        for (auto& [u1, u2] : m) {

                            // Handles edge updates:
                            // if the updated edge is part of the match, we accept it
                            // n1 is the vertex in the candidate graph that corresponds to v1 in the pattern graph
                            // n2 is the vertex in the candidate graph that corresponds to u1 in the pattern
                            // pattern.adjacent checks if v1 and u1 are adjacent in the pattern graph, which they must be for this to be a valid match
                            for (auto& [n1, n2]: changes.edge_updates) {
                                if (v2 == n1 && u2 == n2 && pattern.adjacent(v1, u1)) {
                                    accepted = true;
                                    std::vector<typename GraphType::key_type> order;
                                    std::vector<typename GraphType::key_type> match;
                                    for (auto &[key, value]: m) {
                                        order.push_back(key);
                                        match.push_back(value);
                                    }
                                    DGGML::ComponentMatch<typename GraphType::key_type> inst;
                                    inst.match = match;
                                    inst.order = order;
                                    inst.type = k;
                                    inst.anchor = match[0];
                                    auto res = component_matches.insert(inst);
                                    if (res.second)
                                        validated_components.push_back(res.first->first);
                                    break;
                                }
                                if (accepted) break;
                            }
                        }

                        if (accepted) break;

                    }

                }
            }
        }

        // We now have the commponents that correspond to our rule. (Hopefully)
        // How do I know component_matches is correct?
        for(auto& item : validated_components)
        {
            auto iter = component_matches.find(item);
            cell_list.insert_match(iter);
        }

        std::vector<DGGML::RuleMatch<std::size_t>> accepted_rule_matches;
        std::set<std::size_t> candidate_cells;
        for(auto& item : validated_components)
        {
            auto m1 = component_matches.find(item);

            if (m1 == component_matches.end()) {
                std::cout << "Component match not found for validated component: " << item << std::endl;
                exit(1);
            };

            auto c = cell_list.locate_cell_3d(m1);
            
            // 3D Grid Bounds
            int imin, imax, jmin, jmax, kmin, kmax;
            cell_list.getCells(c, imin, imax, jmin, jmax, kmin, kmax); // Added k bounds

            // 3D Nested Loop
            for (auto i = imin; i < imax; i++) {
                for (auto j = jmin; j < jmax; j++) {
                    for (auto k = kmin; k < kmax; k++) { // Added third loop
                        auto nbr_idx = cell_list.cardinalCellIndex(i, j, k); // 3-arg version
                        candidate_cells.insert(nbr_idx);
                    }
                }
            }
        }

       // FIXME: There is something wrong with candidate cells and cell_list.
       // They are not syncing up for some reason.

       // List through candidate cells
       int matches = 0;
       for(auto& cell : candidate_cells) {
           omp_set_lock(&cell_list.bucket_locks[cell]);
           std::size_t cell_sz = cell_list.data[cell].size();
           omp_unset_lock(&cell_list.bucket_locks[cell]);
           if (cell_sz != 0) matches += 1;
       }

        for(auto& cell : candidate_cells)
        {
            // std::cout << "cell: " << cell << std::endl;
            // Snapshot data[cell] under the bucket lock so we can iterate
            // without holding the lock across the recursive backtracker call.
            std::vector<std::size_t> cell_snapshot;
            omp_set_lock(&cell_list.bucket_locks[cell]);
            cell_snapshot = cell_list.data[cell];
            omp_unset_lock(&cell_list.bucket_locks[cell]);
            for(auto& item : cell_snapshot) {
                auto m1 = component_matches.find(item);
                // Skip components not in this geocell's per-cell map
                // (e.g. components from spatially adjacent d<Dim geocells)
                if (m1 == component_matches.end()) continue;
                // std::cout 
                // Grammar Analysis.rule component is empty?
                for (auto &[name, pattern]: grammar_analysis.rule_component) {
                    int k = 0;
                    std::vector<std::size_t> result;
                    result.resize(pattern.size());
                    if (pattern.size() && m1->second.type == pattern.front()) {
                        result[k] = m1->first;
                        k++;

                        // Finds the cell that the match is in.
                        // This is used for the backtracking algorithm to limit the search space.
                        auto c = cell_list.locate_cell_3d(m1);

                        DGGML3D::incremental_reaction_instance_backtracker(accepted_rule_matches, validated_components,
                                                                  system_graph, component_matches,
                                                                  grammar_analysis, cell_list, name,
                                                                  result, k, pattern, c, reaction_radius);
                    }
                }
            }
        }


       // std::cout << "Number of accepted reaction instances after: " << accepted_rule_matches.size() << "\n";
       // std::cout << std::endl;
        // std::cout << "Done Checking for candidate cells" << std::endl;

        // if(accepted_rule_matches.empty()) {
            // std::cout << "empty rule match" << std::endl;
            // exit(0);
        // }

        // Reinserting rule matches after checking min_dim_phi
        for(auto& inst : accepted_rule_matches)
        {
            // Call the 3D version of min_dim_phi
            auto max_cell = DGGML3D::min_dim_phi(inst, system_graph, geoplex, component_matches);
            if(max_cell == cell_k)
            {
                auto res = rule_matches.insert(inst);
                if(res.second)
                {

                    // std::cout << "adding new rule match " << inst.name << " to rule map\n";
                    // std::cin.get();

                    rule_map.push_back(res.first->first);
                }
            } else
            {
                // Store {max_cell, inst} so cleanup routes the match to the
                // DESTINATION cell's rule_map, not the source cell's.
                geocell_properties.rejected_rule_matches.push_back({max_cell, inst});
            }
        }

    // Listing new matches
    /*
    std::cout << "New rule matches found in geocell " << cell_k << ":\n";
    for(auto& inst : accepted_rule_matches)
    {
        std::cout << "Rule: " << inst.name << ", Components: ";
        for(auto& c : inst.components)
        {
        std::cout << c << " ";
        }
        std::cout << "\n";
    }
    */
    // std::cout << std::endl;
    // std::cin.get();
    // exit(0);

    }


}
#endif //DGGML_INCREMENTALUPDATE_3D_HPP
