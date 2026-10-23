#ifndef DGGML_PATTERN_MATCHING_3D_HPP
#define DGGML_PATTERN_MATCHING_3D_HPP
#include <string>
#include <vector>
#include <set>
#include <algorithm>

#include "ComponentMatchMap.hpp"
#include "RuleMatchMap.hpp"

namespace DGGML3D {

    /**
     * @brief 3D Recursive backtracking for finding full rule instances.
     */
    template<typename SimType>
    void reaction_instance_backtracker(SimType& sim, std::string name, std::vector<std::size_t> &result, int k,
                                       std::vector<std::size_t> &pattern, std::size_t c) {
        if (k == pattern.size()) {
            bool pass = true;
            std::set<std::size_t> all_diff_check;
            for(auto& c1 : result)
            {
                for(auto& c2 : sim.component_matches[c1].match)
                {
                    auto res = all_diff_check.insert(c2);
                    if(!res.second)
                    {
                        pass = false;
                        break;
                    }
                }
                if(!pass) break;
            }
            if(pass) {
                DGGML::RuleMatch<std::size_t> inst;
                inst.name = name;
                if(auto search = sim.grammar_analysis.with_rules.find(name); search != sim.grammar_analysis.with_rules.end())
                    inst.category = "stochastic";
                else if(sim.grammar_analysis.every_rules.find(name) != sim.grammar_analysis.every_rules.end())
                    inst.category = "every";
                else
                    inst.category = "deterministic";
                inst.components = result;
                sim.rule_matches.insert(inst);
            }
        } else {
            // Updated for 3D: Added kmin, kmax and a third loop for the Z-axis
            int imin, imax, jmin, jmax, kmin, kmax;
            sim.cell_list->getCells(c, imin, imax, jmin, jmax, kmin, kmax); 
            
            for (auto i = imin; i < imax; i++) {
                for (auto j = jmin; j < jmax; j++) {
                    for (auto l = kmin; l < kmax; l++) { // New depth loop
                        auto nbr_idx = sim.cell_list->cardinalCellIndex(i, j, l);
                        for (const auto &m2: sim.cell_list->data[nbr_idx]) {
                            bool found = false;
                            for (auto v = 0; v < k; v++) {
                                if (result[v] == m2) found = true;
                            }
                            if (!found && sim.component_matches[m2].type == pattern[k]) {
                                auto a1 = sim.component_matches[result[0]].anchor;
                                auto a2 = sim.component_matches[m2].anchor;
                                auto &p1 = sim.model->system_graph.findNode(a1)->second.getData().position;
                                auto &p2 = sim.model->system_graph.findNode(a2)->second.getData().position;
                                
                                // calculate_distance should handle 3D coordinates automatically
                                auto d = DGGML::calculate_distance(p1, p2); 
                                
                                result[k] = m2;
                                DGGML3D::reaction_instance_backtracker(sim, name, result, k + 1, pattern, c);
                            }
                        }
                    }
                }
            }
        }
    };

    /**
     * @brief 3D Incremental backtracking for updating rule instances near modified nodes.
     * 
     * Finds new matches for rules that may have been created by a local update,
     * by backtracking through the rule pattern starting
     * from a known valid partial match and only searching nearby cells in the reaction grid.
     * This is more efficient than re-checking the entire system for matches after every update.
     */
    template<typename GraphType>
    void incremental_reaction_instance_backtracker(std::vector<DGGML::RuleMatch<std::size_t>>& accepted_rule_matches,
                                                   std::vector<std::size_t>& validated_components,
                                                   GraphType& system_graph,
                                                   DGGML::ComponentMatchMap<std::size_t>& component_matches,
                                                   DGGML::AnalyzedGrammar<GraphType>& grammar_analysis,
                                                   DGGML::CellList<GraphType>& cell_list,
                                                   std::string name, std::vector<std::size_t> &result,
                                                   int k, std::vector<std::size_t> &pattern, std::size_t c,
                                                   double reaction_radius) {
    

        if (k == pattern.size()) {

            // checks if any of the components in this match are 
            // in the validated_components list,
            // which means they were directly updated or adjacent to an update 
            // and thus are automatically valid.
            // If not, we do a more expensive check to ensure all components in the match are distinct and valid.
            bool valid = false;
            for(auto& item : validated_components)
            {
                if(std::find(result.begin(), result.end(), item) != result.end())
                {
                    valid = true;
                    break;
                }
            }

            // checks if all components in the match are distinct, which is a requirement for a valid match.
            bool pass = true;
            if(valid) {
                std::set<std::size_t> all_diff_check;
                for (auto &c1: result) {
                    for (auto &c2: component_matches.at(c1).match) {
                        auto res = all_diff_check.insert(c2);
                        if (!res.second) {
                            pass = false;
                            break;
                        }
                    }
                    if (!pass) break;
                }
            }

            if(valid && pass) {
                DGGML::RuleMatch<std::size_t> inst;
                inst.name = name;
                if(auto search = grammar_analysis.with_rules.find(name); search != grammar_analysis.with_rules.end())
                    inst.category = "stochastic";
                else if(grammar_analysis.every_rules.find(name) != grammar_analysis.every_rules.end())
                    inst.category = "every";
                else
                    inst.category = "deterministic";

                inst.components = result;
                inst.anchor = component_matches.at(result[0]).anchor;
                accepted_rule_matches.push_back(inst);
            }

        } else {

            // Updated for 3D: Added depth range and loop
            int imin, imax, jmin, jmax, kmin, kmax;

            // Get the neighboring cells to search based on the current cell and the reaction radius
            cell_list.getCells(c, imin, imax, jmin, jmax, kmin, kmax);

            for (auto i = imin; i < imax; i++) {
                for (auto j = jmin; j < jmax; j++) {
                    for (auto l = kmin; l < kmax; l++) { // New depth loop
                        auto nbr_idx = cell_list.cardinalCellIndex(i, j, l);

                        // std::cout << "cell_list.data[nbr_idx]: " << cell_list.data[nbr_idx].size() << std::endl;
                        // std::cin.get();
                        // Snapshot nbr_idx bucket under the lock so we can iterate
                        // without holding it across the recursive backtracker call.
                        std::vector<std::size_t> nbr_snapshot;
                        omp_set_lock(&cell_list.bucket_locks[nbr_idx]);
                        nbr_snapshot = cell_list.data[nbr_idx];
                        omp_unset_lock(&cell_list.bucket_locks[nbr_idx]);
                        for (const auto &m2: nbr_snapshot) {

                            bool found = false;
                            for (auto v = 0; v < k; v++) {
                                if (result[v] == m2) found = true;
                            }

                            auto it_m2 = component_matches.find(m2);
                            if (!found && it_m2 != component_matches.end() && it_m2->second.type == pattern[k]) {
                                auto a1 = component_matches.at(result[0]).anchor;
                                auto a2 = it_m2->second.anchor;
                                auto &p1 = system_graph.findNode(a1)->second.getData().position;
                                auto &p2 = system_graph.findNode(a2)->second.getData().position;
                                auto d = DGGML::calculate_distance(p1, p2);

                                // NOTE: This is a limitation we should be aware of.
                                if (d < reaction_radius) {
                                    result[k] = m2;
                                    DGGML3D::incremental_reaction_instance_backtracker(accepted_rule_matches, validated_components,
                                                                              system_graph,component_matches,
                                                                              grammar_analysis, cell_list, name,
                                                                              result, k + 1, pattern, c, reaction_radius);
                                }
                            }

                        }
                    }
                }
            }
        }
    };

}

#endif //DGGML_PATTERN_MATCHING_HPP
