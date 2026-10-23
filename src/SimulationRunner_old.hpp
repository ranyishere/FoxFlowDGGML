#ifndef DGGML_SIMULATIONRUNNER_H
#define DGGML_SIMULATIONRUNNER_H

#include <iostream>
#include <memory>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <chrono>
#include <string>
#include <filesystem>

#include "DggModel.hpp"
#include "YAGL_Graph.hpp"
#include "YAGL_Node.hpp"
#include "YAGL_Algorithms.hpp"
#include "Utlities/VtkWriter.hpp"
#include "ExpandedComplex2D.hpp"
#include "CartesianComplex2D.hpp"

#include "ExpandedComplex3D.hpp"
#include "CartesianComplex3D.hpp"

#include "CellList.hpp"
#include "ComponentMatchMap.hpp"
#include "RuleMatchMap.hpp"
#include "Grammar.h"
#include "Utlities/MathUtils.hpp"

#include "CartesianHashFunctions.hpp"
#include "CartesianHashFunctions3D.hpp"

#include "ApproximateSSA.hpp"
#include "AnalyzedGrammar.hpp"
#include "pattern_matching.hpp"
#include "pattern_matching_3d.hpp"
#include "phi_functions.hpp"
#include "HelperStructs.hpp"

#include "Listeners.hpp"
#include "LivelinessMonitor.hpp"
#include <omp.h>
#include <iomanip>


namespace DGGML {

    struct RunnerTimings {
        double ssa_ms = 0.0;
        double inline_invalidation_ms = 0.0;
        double phase2_cleanup_ms = 0.0;
        double rebuild_ms = 0.0;
        double remap_ms = 0.0;
        double checkpoint_ms = 0.0;

        void reset() { *this = RunnerTimings{}; }

        void print(int step) const {
            double total = ssa_ms + inline_invalidation_ms + phase2_cleanup_ms
                         + rebuild_ms + remap_ms + checkpoint_ms;
            std::cout << std::fixed << std::setprecision(3)
                      << "\n┌─── Step " << step << " Timing Breakdown ─────────────────────────┐\n"
                      << "│  SSA firing:               " << std::setw(10) << ssa_ms              << " ms\n"
                      << "│  Inline invalidation:      " << std::setw(10) << inline_invalidation_ms << " ms\n"
                      << "│  Phase 2 cleanup:          " << std::setw(10) << phase2_cleanup_ms    << " ms\n"
                      << "│  Cell list rebuild:        " << std::setw(10) << rebuild_ms           << " ms\n"
                      << "│  Rule remap:               " << std::setw(10) << remap_ms             << " ms\n"
                      << "│  Checkpoint (VTK):         " << std::setw(10) << checkpoint_ms        << " ms\n"
                      << "│  ─────────────────────────────────────────────────────\n"
                      << "│  TOTAL:                    " << std::setw(10) << total                << " ms\n"
                      << "└───────────────────────────────────────────────────────┘\n";
        }
    };

    template<typename ModelType>
    struct State {
    };


    template<typename ModelType, int Dim = 3>
    class SimulationRunnerOld {

    public:
        using model_type = ModelType;
        using key_type = typename ModelType::key_type;
        using gplex_key_type = typename DGGML::ExpandedComplex2D<>::graph_type::key_type;
        //using data_type = typename ModelType::graph_type;
        //using graph_type = YAGL::Graph<key_type, data_type>;
        using graph_type = typename ModelType::graph_type;
        using node_type = typename graph_type::node_type;
        using rule_key_t = std::size_t;

        using cplex_key_t = typename DGGML::CartesianComplex3D<>::graph_type::key_type;

        bool is_3D = false;

        explicit SimulationRunnerOld(const ModelType& _model, bool is_3D = false)
        : model(std::make_shared<ModelType>(_model))
        {

            if (Dim == 3 || is_3D) {
                this->is_3D = true;
            }

        }

        //TODO: we probably want to make this part of the constructor instead or
        // even better yet, move most of these objects to be external to the class and injected upon
        // construction
        void initialize()
        {
            //Should the model initialize even happen here? or sooner
            std::cout << "Initializing " << model->name << "\n";
            model->initialize();

//            std::cout << "Printing grammar rules...\n";
//            model->gamma.print();

            std::cout << "Performing grammar analysis and building analyzed grammar data structure\n";
            grammar_analysis = AnalyzedGrammar<graph_type>(model->gamma);
            std::cout << "Grammar analysis complete\n";

            //building rule map sets
            for(auto& [key, value] : model->geoplex2D.graph.getNodeSetRef())
                rule_map.insert({key, {}});

            model->checkpoint(0);

            // No change
            compute_single_component_matches();

            // No Change
            set_geocell_propensities();

            // A cell list is used to accelerate the spatial geometric search, but
            // we could use other methods like a bounding volume hierarchy(ArborX), kd-trees etc
            // see books on collision detection like gpu-gems or Real-Time Collision Detection

            cell_list = std::make_unique<CellList<graph_type>>(model->geoplex2D.reaction_grid,
                                            &model->system_graph,
                                            &component_matches, this->is_3D);
            
            cell_list.interaction_radius = model->settings.MAXIMAL_REACTION_RADIUS;


            cell_listener = std::make_unique<DGGML::CellListListener<graph_type>>(cell_list.get());
            cell_list->addListener(cell_listener.get());

            liveliness_monitor = std::make_unique<LivelinessMonitor>();
            model->set_liveliness_monitor(liveliness_monitor);

            // Have a switch here to use 2D or 3D versions of these functions
            compute_all_rule_matches();
            map_rule_matches_to_geocells();
            if (this->is_3D == false) {
                //TODO: deprecate and refactor
                //builds a list of interior geocells to iterate
                // Already have a 3d
                build_bucketsND(bucketsND, model->geoplex2D);
            }
            else {
                // Already have a 3d
                DGGML3D::build_bucketsND(bucketsND, model->geoplex2D);
            }

        }

        void run() {

            //return;
            for(auto i = 0; i <= 2; i++) {
                auto countNd = std::count_if(rule_map.begin(), rule_map.end(),
                                             [&](auto &iter) {
                    return model->geoplex2D.getGraph().findNode(iter.first)->second.getData().type == i && model->geoplex2D.getGraph().findNode(iter.first)->second.getData().interior;
                });
            }

            for(auto i = 1; i <= model->settings.NUM_STEPS; i++)
            {
                RunnerTimings timings;

                // Component inventory
                std::map<std::size_t, int> type_counts;
                for(auto const& [id, match] : component_matches)
                    type_counts[match.type]++;
                std::cout << "\n--- Step " << i << " Component Inventory ---" << std::endl;
                for(auto const& [type, count] : type_counts)
                    std::cout << "  Component Type ID [" << type << "]: " << count << " matches" << std::endl;

                // Dimensional loop
                for(int d = 0; d < Dim + 1; d++)
                {
                    for(auto& bucket : bucketsND[d])
                    {
                        auto k = bucket.first;

                        // --- SSA firing ---
                        {
                            auto t0 = std::chrono::high_resolution_clock::now();
                            approximate_ssa<Dim>(component_matches, grammar_analysis,
                                            rule_map, rule_matches,
                                            model, k, geocell_properties_list[k],
                                            *cell_list, liveliness_monitor.get());
                            timings.ssa_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                        }

                        // --- Inline invalidation propagation ---
                        {
                            auto t0 = std::chrono::high_resolution_clock::now();
                            auto& gp = geocell_properties_list[k];
                            if (!gp.invalidated_components.empty()) {
                                for(auto& [nbr, _val] : model->geoplex2D.graph.getNodeSetRef())
                                {
                                    // if(nbr == k) continue;
                                    std::vector<std::size_t> invalid_rids;
                                    std::vector<std::size_t> invalid_idxs;
                                    for(auto idx = 0; idx < rule_map[nbr].size(); idx++)
                                    {
                                        auto rid = rule_map[nbr][idx];
                                        if(auto search = rule_matches.find(rid); search != rule_matches.end())
                                        {
                                            for(auto& comp : search->second.components)
                                            {
                                                if(gp.invalidated_components.find(comp) != gp.invalidated_components.end())
                                                {
                                                    invalid_rids.push_back(rid);
                                                    invalid_idxs.push_back(idx);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    std::sort(invalid_idxs.begin(), invalid_idxs.end(), std::greater<std::size_t>());
                                    for(int iter = 0; iter < (int)invalid_rids.size(); iter++)
                                    {
                                        rule_matches.erase(invalid_rids[iter]);
                                        rule_map[nbr].erase(rule_map[nbr].begin() + invalid_idxs[iter]);
                                    }
                                }
                                gp.invalidated_components.clear();
                            }
                            timings.inline_invalidation_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                        }
                    }

                    std::cout << d << "D SSA took " << std::fixed << std::setprecision(3)
                              << timings.ssa_ms << " ms (" << bucketsND[d].size() << " buckets)\n";

                    // --- Phase 2 cleanup ---
                    {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for(auto& bucket : bucketsND[d])
                        {
                            auto k = bucket.first;
                            auto& geocell_properties = geocell_properties_list[k];
                            geocell_properties.current_rules_fired = 0;
                            if(!geocell_properties.rejected_rule_matches.empty())
                            {
                                for(auto& [cell_id, rejected_match] : geocell_properties.rejected_rule_matches)
                                {
                                    bool valid = true;
                                    for(auto& comp : rejected_match.components)
                                    {
                                        if(component_matches.find(comp) == component_matches.end())
                                            valid = false;
                                    }
                                    if(valid) {
                                        auto res = rule_matches.insert(rejected_match);
                                        if (res.second)
                                            rule_map[k].push_back(res.first->first);
                                    }
                                }
                                geocell_properties.rejected_rule_matches.clear();
                            }

                            for(auto& [nbr, value] : model->geoplex2D.graph.getNodeSetRef())
                            {
                                std::vector<std::size_t> invalid_rids;
                                std::vector<std::size_t> invalid_idxs;
                                for(auto idx = 0; idx < rule_map[nbr].size(); idx++)
                                {
                                    auto rid = rule_map[nbr][idx];
                                    if(auto search = rule_matches.find(rid); search != rule_matches.end())
                                    {
                                        for(auto& comp : search->second.components)
                                        {
                                            if(geocell_properties.invalidated_components.find(comp) != geocell_properties.invalidated_components.end())
                                            {
                                                invalid_rids.push_back(rid);
                                                invalid_idxs.push_back(idx);
                                                break;
                                            }
                                        }
                                    }
                                }
                                std::sort(invalid_idxs.begin(), invalid_idxs.end(), std::greater<std::size_t>());
                                for(auto iter = 0; iter < invalid_rids.size(); iter++)
                                {
                                    rule_matches.erase(invalid_rids[iter]);
                                    rule_map[nbr].erase(rule_map[nbr].begin()+invalid_idxs[iter]);
                                }
                            }
                            geocell_properties.invalidated_components.clear();
                        }
                        timings.phase2_cleanup_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    }
                }
                // End dimensional loop

                // --- Checkpoint ---
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    model->checkpoint(i);
                    timings.checkpoint_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }
                liveliness_monitor->simulation_steps++;
                std::cout << model->system_graph << "\n";

                // --- Cell list rebuild ---
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cell_list->rebuild();
                    timings.rebuild_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }

                // --- Rule remap ---
                if(!rule_map.empty())
                    for(auto& item : rule_map) item.second.clear();
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    map_rule_matches_to_geocells();
                    timings.remap_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }

                timings.print(i);
            }

            //model->print_metrics();
            //return;

        }

        //private:

        void set_geocell_propensities()
        {
            std::cout << "Setting intial cell propensities to zero\n";
            for(auto& [key, value] : model->geoplex2D.graph.getNodeSetRef())
                geocell_properties_list[key];
        }

        void compute_single_component_matches()
        {
            //single component matches are space invariant, but multi-component matches
            //are not
            std::vector<std::vector<key_type>> match_set;
            //TODO: I think we need to store the ordering or the rooted spanning tree
            for(auto& [k, pattern] : grammar_analysis.unique_components)
            {
                //need to actually get the ordering for the mapping
                auto matches = YAGL::subgraph_isomorphism2(pattern, model->system_graph);
                if(!matches.empty())
                for(auto& item : matches)
                {
                    std::vector<typename ModelType::key_type> order;
                    std::vector<typename ModelType::key_type> match;
                    for(auto& [key, value] : item)
                    {
                        order.push_back(key);
                        match.push_back(value);
                    }
                    ComponentMatch<typename ModelType::key_type> inst;
                    inst.match = match;
                    inst.order = order;
                    inst.type = k;
                    inst.anchor = match[0];
                    component_matches.insert(inst);
                }
                //std::cout << "Found " << matches.size() << " instances\n";
            }

            //for(auto& [k, p] : instances)
//            for(auto& [k, pattern] : grammar_analysis.unique_components)
//            {
//                //std::cout << "Component " << k << " has " << component_matches.count(k) << " instances\n";
//            }

        }

        void compute_all_rule_matches()
        {

            for(auto m1 = component_matches.begin(); m1 != component_matches.end(); m1++)
            {
                for(auto& [name, pattern] : grammar_analysis.rule_component)
                {
                    int k = 0;
                    std::vector<std::size_t> result;
                    result.resize(pattern.size());

                    if(pattern.size() && m1->second.type == pattern.front())
                    {
                        result[k] = m1->first;
                        k++;
                        // auto c = cell_list->locate_cell(m1);

                        if (this->is_3D == false) {
                            auto c = cell_list->locate_cell(m1);
                            DGGML::reaction_instance_backtracker(*this, name, result, k, pattern, c);
                        } else {
                            auto c = cell_list->locate_cell_3d(m1);
                            DGGML3D::reaction_instance_backtracker(*this, name, result, k, pattern, c);
                    }
                }
                }

            }
        }

        void map_rule_matches_to_geocells()
        {

            // depending on the complexity, map anchor nodes to reaction subcells or
            // alternative map reaction instances directly using a computed position
            // or anchor node. We could also use a participation list etc, but for now
            // we've reduced the problem to sorting a whole reaction by an anchor node
            for(auto& [key, inst] : rule_matches)
            {
                //auto max_cell = anchored_phi(inst, model->system_graph, model->geoplex2D);
                if (this->is_3D == false) {
                    auto max_cell = min_dim_phi(inst, model->system_graph, model->geoplex2D, component_matches);
                    rule_map[max_cell].push_back(key);
                } else {
                    auto max_cell = DGGML3D::min_dim_phi(inst, model->system_graph,
                            model->geoplex2D, component_matches);
                    rule_map[max_cell].push_back(key);
                }

            }

        }

        using node_key_type = typename ModelType::key_type;

        std::shared_ptr<ModelType> model;

        std::unique_ptr<CellList<graph_type>> cell_list;

        GeocellPropertiesList<gplex_key_type, node_key_type> geocell_properties_list;
        std::unique_ptr<DGGML::CellListListener<graph_type>> cell_listener;
        std::shared_ptr<LivelinessMonitor> liveliness_monitor;
        AnalyzedGrammar<graph_type> grammar_analysis;
        KeyGenerator<std::size_t> instance_key_gen;

	// These match map essentially act like functions that map a rule key to the components that match the pattern of the rule, and
        RuleMatchMap<node_key_type>  rule_matches;
        ComponentMatchMap<node_key_type> component_matches;

        //TODO: I think rule_maps could steal instances, especially since in this formulation,
        // each instance exists in a shared memory state, the rule_matches map.

        // rule keys for a cell could be a tree/map/set vs vector
        std::map<cplex_key_t, std::vector<rule_key_t>> rule_map; 
        std::map<std::size_t, std::vector<typename ModelType::key_type>> ordering;

        std::map<gplex_key_type, std::vector<key_type>> bucketsND[Dim + 1];

        //DGGML::VtkFileWriter<typename ModelType::graph_type> vtk_writer;
    };

        template<typename ModelType, int Dim = 3>
    class SimulationRunner {

    public:
        using model_type = ModelType;
        using key_type = typename ModelType::key_type;
        using gplex_key_type = typename DGGML::ExpandedComplex2D<>::graph_type::key_type;
        //using data_type = typename ModelType::graph_type;
        //using graph_type = YAGL::Graph<key_type, data_type>;
        using graph_type = typename ModelType::graph_type;
        using node_type = typename graph_type::node_type;
        using rule_key_t = std::size_t;

        using cplex_key_t = typename DGGML::CartesianComplex3D<>::graph_type::key_type;

        bool is_3D = false;

        explicit SimulationRunner(const ModelType& _model, bool is_3D = false)
        : model(std::make_shared<ModelType>(_model))
        {

            if (Dim == 3 || is_3D) {
                this->is_3D = true;
            }

        }

        //TODO: we probably want to make this part of the constructor instead or
        // even better yet, move most of these objects to be external to the class and injected upon
        // construction
        void initialize()
        {
            //Should the model initialize even happen here? or sooner
            std::cout << "Initializing " << model->name << "\n";
            model->initialize();

//            std::cout << "Printing grammar rules...\n";
//            model->gamma.print();

            std::cout << "Performing grammar analysis and building analyzed grammar data structure\n";
            grammar_analysis = AnalyzedGrammar<graph_type>(model->gamma);
            std::cout << "Grammar analysis complete\n";

            //building rule map sets
            for(auto& [key, value] : model->geoplex2D.graph.getNodeSetRef())
                rule_map.insert({key, {}});

            model->checkpoint(0);

            // No change
            compute_single_component_matches();

            // No Change
            set_geocell_propensities();

            // A cell list is used to accelerate the spatial geometric search, but
            // we could use other methods like a bounding volume hierarchy(ArborX), kd-trees etc
            // see books on collision detection like gpu-gems or Real-Time Collision Detection

            auto geo_cell_size = model->settings.CELL_NX * model->settings.CELL_NY *
                (this->is_3D ? model->settings.CELL_NZ : 1);
            
            cell_list = std::make_unique<CellList<graph_type>>(model->geoplex2D.reaction_grid,
                                            &model->system_graph,
                                            &component_matches, this->is_3D, model->settings.MAXIMAL_REACTION_RADIUS
                                        );

            // cell_list->interaction_radius = model->settings.MAXIMAL_REACTION_RADIUS;


            cell_listener = std::make_unique<DGGML::CellListListener<graph_type>>(cell_list.get());
            cell_list->addListener(cell_listener.get());

            liveliness_monitor = std::make_unique<LivelinessMonitor>();
            model->set_liveliness_monitor(liveliness_monitor);


            // Have a switch here to use 2D or 3D versions of these functions
            compute_all_rule_matches();
            map_rule_matches_to_geocells();
            if (this->is_3D == false) {
                //TODO: deprecate and refactor
                //builds a list of interior geocells to iterate
                // Already have a 3d
                build_bucketsND(bucketsND, model->geoplex2D);
            }
            else {
                // Already have a 3d
                DGGML3D::build_bucketsND(bucketsND, model->geoplex2D);
            }

        }

        void run() {

            //return;
            for(auto i = 0; i <= 2; i++) {
                auto countNd = std::count_if(rule_map.begin(), rule_map.end(),
                                             [&](auto &iter) {
                    return model->geoplex2D.getGraph().findNode(iter.first)->second.getData().type == i && model->geoplex2D.getGraph().findNode(iter.first)->second.getData().interior;
                });
            }

            for(auto i = 1; i <= model->settings.NUM_STEPS; i++)
            {
                RunnerTimings timings;

                // Component inventory
                std::map<std::size_t, int> type_counts;
                for(auto const& [id, match] : component_matches)
                    type_counts[match.type]++;
                std::cout << "\n--- Step " << i << " Component Inventory ---" << std::endl;
                for(auto const& [type, count] : type_counts)
                    std::cout << "  Component Type ID [" << type << "]: " << count << " matches" << std::endl;


                    std::map<std::string, int> rule_counts;
                    for(auto const& [id, inst] : rule_matches)
                        rule_counts[inst.name]++;
                    std::cout << "--- Rule Match Inventory ---" << std::endl;
                    for(auto const& [name, count] : rule_counts)
                        std::cout << "  " << name << ": " << count << " matches" << std::endl;

                // Dimensional loop
                for(int d = 0; d < Dim + 1; d++)
                {
                    for(auto& bucket : bucketsND[d])
                    {
                        auto k = bucket.first;

                        // --- SSA firing ---
                        {
                            auto t0 = std::chrono::high_resolution_clock::now();

                            approximate_ssa<Dim>(component_matches, grammar_analysis,
                            rule_map, rule_matches,
                            model, k, geocell_properties_list[k],
                            *cell_list, liveliness_monitor.get(),
                            model->system_graph, model->gen);

                            /* 
                            approximate_ssa<Dim>(component_matches, grammar_analysis,
                                            rule_map, rule_matches,
                                            model, k, geocell_properties_list[k],
                                            *cell_list, liveliness_monitor.get());
                                            */
                            timings.ssa_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                        }

                        // --- Inline invalidation propagation ---
                        {
                            auto t0 = std::chrono::high_resolution_clock::now();
                            auto& gp = geocell_properties_list[k];
                            if (!gp.invalidated_components.empty()) {
                                for(auto& [nbr, _val] : model->geoplex2D.graph.getNodeSetRef())
                                {
                                    // if(nbr == k) continue;
                                    std::vector<std::size_t> invalid_rids;
                                    std::vector<std::size_t> invalid_idxs;
                                    for(auto idx = 0; idx < rule_map[nbr].size(); idx++)
                                    {
                                        auto rid = rule_map[nbr][idx];
                                        if(auto search = rule_matches.find(rid); search != rule_matches.end())
                                        {
                                            for(auto& comp : search->second.components)
                                            {
                                                if(gp.invalidated_components.find(comp) != gp.invalidated_components.end())
                                                {
                                                    invalid_rids.push_back(rid);
                                                    invalid_idxs.push_back(idx);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    std::sort(invalid_idxs.begin(), invalid_idxs.end(), std::greater<std::size_t>());
                                    for(int iter = 0; iter < (int)invalid_rids.size(); iter++)
                                    {
                                        rule_matches.erase(invalid_rids[iter]);
                                        rule_map[nbr].erase(rule_map[nbr].begin() + invalid_idxs[iter]);
                                    }
                                }
                                gp.invalidated_components.clear();
                            }
                            timings.inline_invalidation_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0).count();
                        }
                    }

                    std::cout << d << "D SSA took " << std::fixed << std::setprecision(3)
                              << timings.ssa_ms << " ms (" << bucketsND[d].size() << " buckets)\n";

                    // --- Phase 2 cleanup ---
                    {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for(auto& bucket : bucketsND[d])
                        {
                            auto k = bucket.first;
                            auto& geocell_properties = geocell_properties_list[k];
                            geocell_properties.current_rules_fired = 0;
                            if(!geocell_properties.rejected_rule_matches.empty())
                            {
                                for(auto& [cell_id, rejected_match] : geocell_properties.rejected_rule_matches)
                                {
                                    bool valid = true;
                                    for(auto& comp : rejected_match.components)
                                    {
                                        if(component_matches.find(comp) == component_matches.end())
                                            valid = false;
                                    }
                                    if(valid) {
                                        auto res = rule_matches.insert(rejected_match);
                                        if (res.second)
                                            rule_map[k].push_back(res.first->first);
                                    }
                                }
                                geocell_properties.rejected_rule_matches.clear();
                            }

                            // Removing stale rule matches that were invalidated by the SSA firing and
                            // inline invalidation phases. We check the neighbors of the geocell because rule matches
                            // are mapped to geocells based on an anchor component,
                            // but they can be invalidated by changes to any of their components,
                            // which may be located in neighboring geocells.
                            for(auto& [nbr, value] : model->geoplex2D.graph.getNodeSetRef())
                            {
                                std::vector<std::size_t> invalid_rids;
                                std::vector<std::size_t> invalid_idxs;
                                for(auto idx = 0; idx < rule_map[nbr].size(); idx++)
                                {
                                    auto rid = rule_map[nbr][idx];
                                    if(auto search = rule_matches.find(rid); search != rule_matches.end())
                                    {
                                        for(auto& comp : search->second.components)
                                        {
                                            if(geocell_properties.invalidated_components.find(comp) != geocell_properties.invalidated_components.end())
                                            {
                                                invalid_rids.push_back(rid);
                                                invalid_idxs.push_back(idx);
                                                break;
                                            }
                                        }
                                    }
                                }
                                std::sort(invalid_idxs.begin(), invalid_idxs.end(), std::greater<std::size_t>());
                                for(auto iter = 0; iter < invalid_rids.size(); iter++)
                                {
                                    rule_matches.erase(invalid_rids[iter]);
                                    rule_map[nbr].erase(rule_map[nbr].begin()+invalid_idxs[iter]);
                                }
                            }
                            geocell_properties.invalidated_components.clear();
                        }
                        timings.phase2_cleanup_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    }
                }
                // End dimensional loop

                // --- Checkpoint ---
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    model->checkpoint(i);
                    timings.checkpoint_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }
                liveliness_monitor->simulation_steps++;
                std::cout << model->system_graph << "\n";

                // --- Cell list rebuild ---
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cell_list->rebuild();
                    timings.rebuild_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }

                // --- Rule remap ---
                if(!rule_map.empty())
                    for(auto& item : rule_map) item.second.clear();
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    map_rule_matches_to_geocells();
                    timings.remap_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }

                timings.print(i);
            }

            //model->print_metrics();
            //return;

        }

        //private:

        void set_geocell_propensities()
        {
            std::cout << "Setting intial cell propensities to zero\n";
            for(auto& [key, value] : model->geoplex2D.graph.getNodeSetRef())
                geocell_properties_list[key];
        }

        void compute_single_component_matches()
        {
            //single component matches are space invariant, but multi-component matches
            //are not
            std::vector<std::vector<key_type>> match_set;
            //TODO: I think we need to store the ordering or the rooted spanning tree
            for(auto& [k, pattern] : grammar_analysis.unique_components)
            {
                //need to actually get the ordering for the mapping
                auto matches = YAGL::subgraph_isomorphism2(pattern, model->system_graph);
                if(!matches.empty())
                for(auto& item : matches)
                {
                    std::vector<typename ModelType::key_type> order;
                    std::vector<typename ModelType::key_type> match;
                    for(auto& [key, value] : item)
                    {
                        order.push_back(key);
                        match.push_back(value);
                    }
                    ComponentMatch<typename ModelType::key_type> inst;
                    inst.match = match;
                    inst.order = order;
                    inst.type = k;
                    inst.anchor = match[0];
                    component_matches.insert(inst);
                }
                //std::cout << "Found " << matches.size() << " instances\n";
            }

            //for(auto& [k, p] : instances)
//            for(auto& [k, pattern] : grammar_analysis.unique_components)
//            {
//                //std::cout << "Component " << k << " has " << component_matches.count(k) << " instances\n";
//            }

        }

        void compute_all_rule_matches()
        {

            for(auto m1 = component_matches.begin(); m1 != component_matches.end(); m1++)
            {
                for(auto& [name, pattern] : grammar_analysis.rule_component)
                {
                    int k = 0;
                    std::vector<std::size_t> result;
                    result.resize(pattern.size());

                    if(pattern.size() && m1->second.type == pattern.front())
                    {
                        result[k] = m1->first;
                        k++;
                        // auto c = cell_list->locate_cell(m1);

                        if (this->is_3D == false) {
                            auto c = cell_list->locate_cell(m1);
                            DGGML::reaction_instance_backtracker(*this, name, result, k, pattern, c);
                        } else {
                            auto c = cell_list->locate_cell_3d(m1);
                            DGGML3D::reaction_instance_backtracker(*this, name, result, k, pattern, c);
                    }
                }
                }

            }
        }

        void map_rule_matches_to_geocells()
        {

            // depending on the complexity, map anchor nodes to reaction subcells or
            // alternative map reaction instances directly using a computed position
            // or anchor node. We could also use a participation list etc, but for now
            // we've reduced the problem to sorting a whole reaction by an anchor node
            for(auto& [key, inst] : rule_matches)
            {
                //auto max_cell = anchored_phi(inst, model->system_graph, model->geoplex2D);
                if (this->is_3D == false) {
                    auto max_cell = min_dim_phi(inst, model->system_graph, model->geoplex2D, component_matches);
                    rule_map[max_cell].push_back(key);
                } else {
                    auto max_cell = DGGML3D::min_dim_phi(inst, model->system_graph,
                            model->geoplex2D, component_matches);
                    rule_map[max_cell].push_back(key);
                }

            }

        }

        using node_key_type = typename ModelType::key_type;

        std::shared_ptr<ModelType> model;

        std::unique_ptr<CellList<graph_type>> cell_list;

        GeocellPropertiesList<gplex_key_type, node_key_type> geocell_properties_list;
        std::unique_ptr<DGGML::CellListListener<graph_type>> cell_listener;
        std::shared_ptr<LivelinessMonitor> liveliness_monitor;
        AnalyzedGrammar<graph_type> grammar_analysis;
        KeyGenerator<std::size_t> instance_key_gen;

	// These match map essentially act like functions that map a rule key to the components that match the pattern of the rule, and
        RuleMatchMap<node_key_type>  rule_matches;
        ComponentMatchMap<node_key_type> component_matches;

        //TODO: I think rule_maps could steal instances, especially since in this formulation,
        // each instance exists in a shared memory state, the rule_matches map.

        // rule keys for a cell could be a tree/map/set vs vector
        std::map<cplex_key_t, std::vector<rule_key_t>> rule_map; 
        std::map<std::size_t, std::vector<typename ModelType::key_type>> ordering;

        std::map<gplex_key_type, std::vector<key_type>> bucketsND[Dim + 1];

        //DGGML::VtkFileWriter<typename ModelType::graph_type> vtk_writer;
    };



    template<typename ModelType, int Dim = 3>
    class SimulationRunnerParallel {

    public:
        using model_type = ModelType;
        using key_type = typename ModelType::key_type;
        using gplex_key_type = typename DGGML::ExpandedComplex2D<>::graph_type::key_type;
        //using data_type = typename ModelType::graph_type;
        //using graph_type = YAGL::Graph<key_type, data_type>;
        using graph_type = typename ModelType::graph_type;
        using node_type = typename graph_type::node_type;
        using rule_key_t = std::size_t;

        using cplex_key_t = typename DGGML::CartesianComplex3D<>::graph_type::key_type;
        using node_key_type = typename ModelType::key_type;

        bool is_3D = false;

        explicit SimulationRunnerParallel(const ModelType& _model, bool is_3D = false)
        : model(std::make_shared<ModelType>(_model))
        {

            if (Dim == 3 || is_3D) {
                this->is_3D = true;
            }

        }

        //TODO: we probably want to make this part of the constructor instead or
        // even better yet, move most of these objects to be external to the class and injected upon
        // construction
        void initialize()
        {
            //Should the model initialize even happen here? or sooner
            std::cout << "Initializing " << model->name << "\n";
            model->initialize();

//            std::cout << "Printing grammar rules...\n";
//            model->gamma.print();

            std::cout << "Performing grammar analysis and building analyzed grammar data structure\n";
            grammar_analysis = AnalyzedGrammar<graph_type>(model->gamma);
            std::cout << "Grammar analysis complete\n";

            //building rule map sets
            for(auto& [key, value] : model->geoplex2D.graph.getNodeSetRef())
                rule_map.insert({key, {}});

            model->checkpoint(0);

            // No change
            compute_single_component_matches();

            // No Change
            set_geocell_propensities();

            // A cell list is used to accelerate the spatial geometric search, but
            // we could use other methods like a bounding volume hierarchy(ArborX), kd-trees etc
            // see books on collision detection like gpu-gems or Real-Time Collision Detection

            cell_list = std::make_unique<CellList<graph_type>>(model->geoplex2D.reaction_grid,
                                            &model->system_graph,
                                            &component_matches, this->is_3D);


            cell_listener = std::make_unique<DGGML::CellListListener<graph_type>>(cell_list.get());
            cell_list->addListener(cell_listener.get());

            liveliness_monitor = std::make_unique<LivelinessMonitor>();
            model->set_liveliness_monitor(liveliness_monitor);

            // Have a switch here to use 2D or 3D versions of these functions
            compute_all_rule_matches();

            // Pre-size hash maps to prevent rehash during the parallel phase.
            // With omp critical(rewrite_phase) serializing all writes, concurrent reads
            // from the propensity loop access disjoint buckets (different geocell keys).
            // No rehash during parallel execution keeps those bucket pointers stable.
            component_matches.matches.reserve(component_matches.size() * 4);
            rule_matches.matches.reserve(rule_matches.size() * 4);
            map_rule_matches_to_geocells();
            if (this->is_3D == false) {
                //TODO: deprecate and refactor
                //builds a list of interior geocells to iterate
                // Already have a 3d
                build_bucketsND(bucketsND, model->geoplex2D);
            }
            else {
                // Already have a 3d
                DGGML3D::build_bucketsND(bucketsND, model->geoplex2D);
            }

            // Initialize per-cell data structures for d=0 (3D volumetric) parallel phase.
            // d=0 rules are single-component diffusion rules: each component's anchor
            // uniquely identifies its owning cell, so per-cell maps are fully disjoint.
            for (auto& [k, _] : bucketsND[0]) {
                component_matches_by_cell[k];          // default-construct
                rule_matches_by_cell[k];               // default-construct
                gen_by_cell[k] = KeyGenerator<node_key_type>(0);  // placeholder; reset in distribute
            }

            // Initialize per-cell OMP locks for ALL dimensions.
            // All geocells are spatially independent, so all can run in parallel.
            // Each thread holds lock[k] for its entire approximate_ssa call, meaning
            // all graph reads (propensity) and writes (do_rewrite) for cell k are
            // serialised relative to any other thread that might access cell k's nodes.
            // Cells in different spatial regions own disjoint node sets, so threads
            // on different cells are truly race-free on system_graph access.
            // Destroy any previously initialised locks first (e.g. if initialize() is re-called).
            /*
            for (auto& [k, lk] : cell_locks)
                omp_destroy_lock(&lk);
            cell_locks.clear();
            std::size_t total_cells = 0;
            for (int d = 0; d <= Dim; d++) total_cells += bucketsND[d].size();
            cell_locks.reserve(total_cells * 2);  // prevent rehash during inserts
            for (int d = 0; d <= Dim; d++) {
                for (auto& [k, _] : bucketsND[d]) {
                    omp_init_lock(&cell_locks[k]);
                }
            }
                */

        }

        void run() {

            for(auto i = 0; i <= 2; i++) {
                auto countNd = std::count_if(rule_map.begin(), rule_map.end(),
                                             [&](auto &iter) {
                    return model->geoplex2D.getGraph().findNode(iter.first)->second.getData().type == i && model->geoplex2D.getGraph().findNode(iter.first)->second.getData().interior;
                });
            }

            for(auto i = 1; i <= model->settings.NUM_STEPS; i++)
            {
                RunnerTimings timings;

                // Component inventory
                std::map<std::size_t, int> type_counts;
                for(auto const& [id, match] : component_matches)
                    type_counts[match.type]++;
                std::cout << "\n--- Step " << i << " Component Inventory ---" << std::endl;
                for(auto const& [type, count] : type_counts)
                    std::cout << "  Component Type ID [" << type << "]: " << count << " matches" << std::endl;

                // Dimensional loop
                for(int d = 0; d < Dim + 1; d++)
                {
                    // Build a random-access vector so OpenMP can index into it
                    // this contains all the geocells
                    // of the current dimension that we need to iterate over for SSA propensity and firing
                    // We are essentially just making a copy here to be able to
                    // parallelize the loop with OpenMP, since std::map doesn't support random access iterators
                    std::vector<std::pair<gplex_key_type, std::vector<key_type>>> buckets_vec(
                        bucketsND[d].begin(), bucketsND[d].end());

                    // Build per-cell read-only snapshots of rule_matches and component_matches.
                    // During the parallel SSA propensity loop each thread reads only from its
                    // own small private copy, keeping those hot cache lines out of shared state.
                    // All entries are pre-populated here in the serial region (no race), and
                    // rebuilt inside omp critical after each do_rewrite inside approximate_ssa.
                    using rm_snap_t = std::unordered_map<rule_key_t, DGGML::RuleMatch<node_key_type>>;
                    using cm_snap_t = DGGML::ComponentMatchMap<node_key_type>;
                    std::unordered_map<gplex_key_type, rm_snap_t>  rm_snaps;
                    std::unordered_map<gplex_key_type, cm_snap_t>  cm_snaps;
                    rm_snaps.reserve(buckets_vec.size() * 2);
                    cm_snaps.reserve(buckets_vec.size() * 2);
                    for (auto& [k, _] : buckets_vec) {
                        auto& rm_k = rm_snaps[k];
                        auto& cm_k = cm_snaps[k];
                        auto rm_it = rule_map.find(k);
                        if (rm_it == rule_map.end()) continue;
                        for (auto rid : rm_it->second) {
                            auto rit = rule_matches.matches.find(rid);
                            if (rit == rule_matches.matches.end()) continue;
                            rm_k[rid] = rit->second;
                            for (auto cid : rit->second.components) {
                                auto cit = component_matches.matches.find(cid);
                                if (cit != component_matches.matches.end())
                                    cm_k.matches[cid] = cit->second;
                            }
                        }
                    }

                    // --- SSA firing ---
                    {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        if (d == 0) {
                            // Worklist / optimistic concurrency for volumetric cells.
                            // 1. Snapshot global maps into per-cell isolated copies.
                            // 2. Each thread rewrites its own copy with no locks.
                            // 3. After the parallel phase, detect conflicts (same
                            //    component invalidated by two cells) and roll back
                            //    the losing cell.  Merge winning cells back to global.
                            distribute_to_per_cell();

                            // Save pre-parallel per-cell state for conflict rollback.
                            auto comp_snap = component_matches_by_cell;
                            auto rule_snap = rule_matches_by_cell;
                            std::unordered_map<gplex_key_type, std::vector<rule_key_t>> rule_map_snap;
                            for (auto& [k, _] : bucketsND[0])
                                rule_map_snap[k] = rule_map.at(k);

                            #pragma omp parallel for schedule(dynamic)
                            for(int bi = 0; bi < (int)buckets_vec.size(); bi++)
                            {
                                auto k = buckets_vec[bi].first;
                                // Each thread writes to its own per-cell maps; no lock needed.
                                approximate_ssa<Dim>(component_matches_by_cell.at(k), grammar_analysis,
                                                rule_map, rule_matches_by_cell.at(k),
                                                model, k, geocell_properties_list.at(k),
                                                *cell_list, liveliness_monitor.get(),
                                                model->system_graph, gen_by_cell.at(k),
                                                rm_snaps.at(k), cm_snaps.at(k));
                            }

                            // Conflict detection: if two cells both invalidated the same
                            // component, one of them saw stale state.  Roll it back so
                            // the event retries next step via rejected_rule_matches.
                            // NOTE: This should not happen! This is just a precaution to capture
                            // any bugs in the parallel SSA implementation.  In theory, with
                            // proper locking of shared state in approximate_ssa, this should be impossible, because
                            // each cell's SSA should be fully isolated from any other cell's SSA,
                            // with no shared state access at all. 
                            // If we do see conflicts here,
                            // it means there's a bug in the parallel SSA that's allowing shared state to be accessed without proper locking, which is causing non
                            std::unordered_map<std::size_t, gplex_key_type> comp_owner;
                            std::unordered_set<gplex_key_type> conflicted_cells;
                            for (auto& [k, _] : bucketsND[0]) {
                                for (auto cid : geocell_properties_list[k].invalidated_components) {
                                    auto [it, inserted] = comp_owner.emplace(cid, k);
                                    if (!inserted && it->second != k)
                                        conflicted_cells.insert(k);
                                }
                            }
                            if (!conflicted_cells.empty()) {
                                std::cout << "[worklist] " << conflicted_cells.size()
                                          << " conflicted geocell(s) rolled back.\n";
                                for (auto kc : conflicted_cells) {
                                    component_matches_by_cell[kc] = comp_snap[kc];
                                    rule_matches_by_cell[kc]      = rule_snap[kc];
                                    rule_map.at(kc)               = rule_map_snap[kc];
                                    geocell_properties_list[kc].invalidated_components.clear();
                                    geocell_properties_list[kc].rejected_rule_matches.clear();
                                }
                                exit(1);
                            }

                            // Merge per-cell post-rewrite state back into global maps.
                            merge_from_per_cell();

                        } else {
                            // d>0 lower-dimensional cells are few; run serially with
                            // global maps to avoid data races (no per-cell maps for d>0).
                            for (auto& [k, _] : bucketsND[d]) {
                                approximate_ssa<Dim>(component_matches, grammar_analysis,
                                                rule_map, rule_matches,
                                                model, k, geocell_properties_list.at(k),
                                                *cell_list, liveliness_monitor.get(),
                                                model->system_graph, model->gen,
                                                rm_snaps.at(k), cm_snaps.at(k));
                            }
                        }
                        timings.ssa_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    }

                    std::cout << d << "D SSA took " << std::fixed << std::setprecision(3)
                              << timings.ssa_ms << " ms (" << bucketsND[d].size() << " buckets)\n";

                    // --- Phase 2 cleanup ---
                    {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for(auto& bucket : bucketsND[d])
                        {
                            auto k = bucket.first;
                            auto& geocell_properties = geocell_properties_list[k];
                            geocell_properties.current_rules_fired = 0;

                            if(!geocell_properties.rejected_rule_matches.empty())
                            {
                                for(auto& [cell_id, rejected_match] : geocell_properties.rejected_rule_matches)
                                {
                                    bool valid = true;
                                    for(auto& comp : rejected_match.components)
                                        if(component_matches.find(comp) == component_matches.end())
                                            valid = false;
                                    if(valid) {
                                        auto res = rule_matches.insert(rejected_match);
                                        if (res.second)
                                            // Route to DESTINATION cell (cell_id = max_cell set
                                            // in find_new_matches), not the source cell k.
                                            rule_map[cell_id].push_back(res.first->first);
                                    }
                                }
                                geocell_properties.rejected_rule_matches.clear();
                            }

                            for(auto& [nbr, value] : model->geoplex2D.graph.getNodeSetRef())
                            {
                                std::vector<std::size_t> invalid_rids;
                                std::vector<std::size_t> invalid_idxs;
                                for(auto idx = 0; idx < rule_map[nbr].size(); idx++)
                                {
                                    auto rid = rule_map[nbr][idx];
                                    if(auto search = rule_matches.find(rid); search != rule_matches.end())
                                    {
                                        for(auto& comp : search->second.components)
                                        {
                                            if(geocell_properties.invalidated_components.find(comp)
                                                != geocell_properties.invalidated_components.end())
                                            {
                                                invalid_rids.push_back(rid);
                                                invalid_idxs.push_back(idx);
                                                break;
                                            }
                                        }
                                    }
                                }
                                std::sort(invalid_idxs.begin(), invalid_idxs.end(), std::greater<std::size_t>());
                                for(int iter = 0; iter < (int)invalid_rids.size(); iter++)
                                {
                                    rule_matches.erase(invalid_rids[iter]);
                                    rule_map[nbr].erase(rule_map[nbr].begin() + invalid_idxs[iter]);
                                }
                            }
                            geocell_properties.invalidated_components.clear();
                        }
                        timings.phase2_cleanup_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                    }
                }
                // End dimensional loop

                // --- Checkpoint ---
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    model->checkpoint(i);
                    timings.checkpoint_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }
                liveliness_monitor->simulation_steps++;

                // Remove if not needed
                std::cout << model->system_graph << "\n";

                // --- Cell list rebuild ---
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    cell_list->rebuild();
                    timings.rebuild_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }

                // --- Rule remap ---
                if(!rule_map.empty())
                    for(auto& item : rule_map) item.second.clear();
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    map_rule_matches_to_geocells();
                    timings.remap_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                }

                timings.print(i);
            }

        }

        //private:

        void set_geocell_propensities()
        {
            std::cout << "Setting intial cell propensities to zero\n";
            for(auto& [key, value] : model->geoplex2D.graph.getNodeSetRef())
                geocell_properties_list[key];
        }

        void compute_single_component_matches()
        {
            //single component matches are space invariant, but multi-component matches
            //are not
            std::vector<std::vector<key_type>> match_set;
            //TODO: I think we need to store the ordering or the rooted spanning tree
            for(auto& [k, pattern] : grammar_analysis.unique_components)
            {
                //need to actually get the ordering for the mapping
                auto matches = YAGL::subgraph_isomorphism2(pattern, model->system_graph);
                if(!matches.empty())
                for(auto& item : matches)
                {
                    std::vector<typename ModelType::key_type> order;
                    std::vector<typename ModelType::key_type> match;
                    for(auto& [key, value] : item)
                    {
                        order.push_back(key);
                        match.push_back(value);
                    }
                    ComponentMatch<typename ModelType::key_type> inst;
                    inst.match = match;
                    inst.order = order;
                    inst.type = k;
                    inst.anchor = match[0];
                    component_matches.insert(inst);
                }
                //std::cout << "Found " << matches.size() << " instances\n";
            }

            //for(auto& [k, p] : instances)
//            for(auto& [k, pattern] : grammar_analysis.unique_components)
//            {
//                //std::cout << "Component " << k << " has " << component_matches.count(k) << " instances\n";
//            }

        }

        void compute_all_rule_matches()
        {

            for(auto m1 = component_matches.begin(); m1 != component_matches.end(); m1++)
            {
                for(auto& [name, pattern] : grammar_analysis.rule_component)
                {
                    int k = 0;
                    std::vector<std::size_t> result;
                    result.resize(pattern.size());

                    if(pattern.size() && m1->second.type == pattern.front())
                    {
                        result[k] = m1->first;
                        k++;
                        // auto c = cell_list->locate_cell(m1);

                        if (this->is_3D == false) {
                            auto c = cell_list->locate_cell(m1);
                            DGGML::reaction_instance_backtracker(*this, name, result, k, pattern, c);
                        } else {
                            auto c = cell_list->locate_cell_3d(m1);
                            DGGML3D::reaction_instance_backtracker(*this, name, result, k, pattern, c);
                    }
                }
                }

            }
        }

        void map_rule_matches_to_geocells()
        {

            // depending on the complexity, map anchor nodes to reaction subcells or
            // alternative map reaction instances directly using a computed position
            // or anchor node. We could also use a participation list etc, but for now
            // we've reduced the problem to sorting a whole reaction by an anchor node
            for(auto& [key, inst] : rule_matches)
            {
                //auto max_cell = anchored_phi(inst, model->system_graph, model->geoplex2D);
                if (this->is_3D == false) {
                    auto max_cell = min_dim_phi(inst, model->system_graph, model->geoplex2D, component_matches);
                    rule_map[max_cell].push_back(key);
                } else {
                    auto max_cell = DGGML3D::min_dim_phi(inst, model->system_graph,
                            model->geoplex2D, component_matches);
                    rule_map[max_cell].push_back(key);
                }

            }

        }

        // Returns the d=Dim geocell key that owns node nkey, or 0 if not found.
        gplex_key_type geocell_for_anchor(node_key_type nkey) const
        {
            auto it = model->system_graph.findNode(nkey);
            if (it == model->system_graph.node_list_end()) return 0;
            auto& pos = it->second.getData().position;
            int ic, jc, kc;
            model->geoplex2D.reaction_grid.locatePoint(pos[0], pos[1], pos[2], ic, jc, kc);
            auto cardinal = model->geoplex2D.reaction_grid.cardinalCellIndex(ic, jc, kc);
            if (cardinal < 0 || (std::size_t)cardinal >= model->geoplex2D.cell_label.size()) return 0;
            return model->geoplex2D.cell_label[cardinal];
        }

        // Copy d=0 component/rule matches into per-cell maps before the parallel phase.
        // Also snapshots which IDs belong to d=0 so merge_from_per_cell can remove them.
        void distribute_to_per_cell()
        {
            // 1. Clear per-cell maps
            for (auto& [k, _] : bucketsND[0]) {
                component_matches_by_cell.at(k).matches.clear();
                rule_matches_by_cell.at(k).matches.clear();
            }

            // 2. Snapshot d=0 component IDs and distribute to per-cell maps
            dim_comp_ids_before.clear();
            for (auto& [comp_key, match] : component_matches) {
                auto cell_k = geocell_for_anchor(match.anchor);
                if (bucketsND[0].count(cell_k)) {
                    dim_comp_ids_before.insert(comp_key);
                    component_matches_by_cell.at(cell_k).matches.insert({comp_key, match});
                }
            }

            // 3. Snapshot d=0 rule IDs and distribute to per-cell maps
            dim_rule_ids_before.clear();
            for (auto& [cell_k, rule_ids] : rule_map) {
                if (!bucketsND[0].count(cell_k)) continue;
                for (auto rid : rule_ids) {
                    dim_rule_ids_before.insert(rid);
                    auto it = rule_matches.find(rid);
                    if (it != rule_matches.end())
                        rule_matches_by_cell.at(cell_k).matches.insert({rid, it->second});
                }
            }

            // 4. Assign non-overlapping key generator ranges to each per-cell generator
            {
                auto base_key = model->gen.current_key;
                node_key_type idx = 0;
                for (auto& [cell_k, _] : bucketsND[0]) {
                    gen_by_cell.at(cell_k) = KeyGenerator<node_key_type>(
                        base_key + idx * KEY_RANGE_PER_CELL);
                    idx++;
                }
                model->gen.current_key = base_key + (node_key_type)bucketsND[0].size() * KEY_RANGE_PER_CELL;
            }

            // 4b. Assign non-overlapping key ranges to per-cell component match key_gens.
            // find_new_matches generates new component IDs via the per-cell map's key_gen;
            // without disjoint ranges across cells, all cells start at 0 and collide on merge.
            {
                auto base_key = (node_key_type)component_matches.key_gen.current_key;
                node_key_type idx = 0;
                for (auto& [cell_k, _] : bucketsND[0]) {
                    component_matches_by_cell.at(cell_k).key_gen.current_key =
                        base_key + idx * KEY_RANGE_PER_CELL;
                    idx++;
                }
                component_matches.key_gen.current_key =
                    base_key + (node_key_type)bucketsND[0].size() * KEY_RANGE_PER_CELL;
            }

            // 4c. Same for per-cell rule match key_gens.
            {
                auto base_key = (rule_key_t)rule_matches.key_gen.current_key;
                rule_key_t idx = 0;
                for (auto& [cell_k, _] : bucketsND[0]) {
                    rule_matches_by_cell.at(cell_k).key_gen.current_key =
                        base_key + idx * (rule_key_t)KEY_RANGE_PER_CELL;
                    idx++;
                }
                rule_matches.key_gen.current_key =
                    base_key + (rule_key_t)bucketsND[0].size() * (rule_key_t)KEY_RANGE_PER_CELL;
            }
        }

        // After the parallel phase, sync per-cell match maps back to global maps.
        void merge_from_per_cell()
        {
            // Remove pre-distribute d=Dim components from global map
            for (auto key : dim_comp_ids_before)
                component_matches.matches.erase(key);

            // Remove pre-distribute d=Dim rules from global map
            for (auto rid : dim_rule_ids_before)
                rule_matches.matches.erase(rid);

            // Sync updated per-cell component maps back to global
            for (auto& [cell_k, _] : bucketsND[0]) {
                for (auto& [comp_key, match] : component_matches_by_cell.at(cell_k).matches)
                    component_matches.matches.insert({comp_key, match});
            }

            // Sync updated per-cell rule maps back to global
            // rule_map[cell_k] was updated in-place during the parallel phase by
            // perform_invalidations and find_new_matches; use it to identify valid IDs.
            for (auto& [cell_k, _] : bucketsND[0]) {
                for (auto rid : rule_map.at(cell_k)) {
                    auto it = rule_matches_by_cell.at(cell_k).matches.find(rid);
                    if (it != rule_matches_by_cell.at(cell_k).matches.end())
                        rule_matches.matches.insert({rid, it->second});
                }
            }

            // Advance model->gen past the highest key produced by any per-cell generator
            for (auto& [cell_k, g] : gen_by_cell)
                if (g.current_key > model->gen.current_key)
                    model->gen.current_key = g.current_key;

            // Advance component/rule key_gens past the highest key produced by per-cell gens.
            // (distribute_to_per_cell already set global past ALL per-cell ranges, but we
            //  tighten it here in case cells consumed far fewer than KEY_RANGE_PER_CELL keys.)
            for (auto& [cell_k, cm] : component_matches_by_cell)
                if (cm.key_gen.current_key > component_matches.key_gen.current_key)
                    component_matches.key_gen.current_key = cm.key_gen.current_key;
            for (auto& [cell_k, rm] : rule_matches_by_cell)
                if (rm.key_gen.current_key > rule_matches.key_gen.current_key)
                    rule_matches.key_gen.current_key = rm.key_gen.current_key;
        }

        std::shared_ptr<ModelType> model;

        std::unique_ptr<CellList<graph_type>> cell_list;

        GeocellPropertiesList<gplex_key_type, node_key_type> geocell_properties_list;
        std::unique_ptr<DGGML::CellListListener<graph_type>> cell_listener;
        std::shared_ptr<LivelinessMonitor> liveliness_monitor;
        AnalyzedGrammar<graph_type> grammar_analysis;
        KeyGenerator<std::size_t> instance_key_gen;

	// These match map essentially act like functions that map a rule key to the components that match the pattern of the rule, and
        RuleMatchMap<node_key_type>  rule_matches;
        ComponentMatchMap<node_key_type> component_matches;

        //TODO: I think rule_maps could steal instances, especially since in this formulation,
        // each instance exists in a shared memory state, the rule_matches map.

        // rule keys for a cell could be a tree/map/set vs vector
        std::map<cplex_key_t, std::vector<rule_key_t>> rule_map; 
        std::map<std::size_t, std::vector<typename ModelType::key_type>> ordering;

        std::map<gplex_key_type, std::vector<key_type>> bucketsND[Dim + 1];

        // Per-cell isolated match maps for the d=0 parallel SSA phase (kept for reference).
        std::unordered_map<gplex_key_type, ComponentMatchMap<node_key_type>> component_matches_by_cell;
        std::unordered_map<gplex_key_type, RuleMatchMap<node_key_type>>      rule_matches_by_cell;
        std::unordered_map<gplex_key_type, KeyGenerator<node_key_type>>      gen_by_cell;
        static constexpr node_key_type KEY_RANGE_PER_CELL = 100000;
        std::unordered_set<node_key_type> dim_comp_ids_before;
        std::unordered_set<rule_key_t>    dim_rule_ids_before;

        // Per-cell OMP locks for d=0 parallel SSA.
        // One lock per 3D volumetric geocell: structural rewrites for cell k acquire
        // cell_locks[k], allowing rewrites in *different* cells to overlap in time.
        // The global maps (component_matches, rule_matches, rule_map) are pre-reserved
        // so no rehash occurs during parallel inserts to different keys.
        // std::unordered_map<gplex_key_type, omp_lock_t> cell_locks;

        //DGGML::VtkFileWriter<typename ModelType::graph_type> vtk_writer;
    };

};

#endif //DGGML_SIMULATIONRUNNER_H
