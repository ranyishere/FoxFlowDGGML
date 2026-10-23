#include "Listeners.hpp"
#include "HelperStructs.hpp"
#include "IncrementalUpdate.hpp"


#ifndef DGGML_LIVELINESSMONITOR_HPP
#define DGGML_LIVELINESSMONITOR_HPP

namespace DGGML {

    /**
    * cell_list - Bins components into spatial cells for fast loopup
    *    PropensityListener* propensity_listener = nullptr;
    *    RuleFireListener* rule_fire_listener = nullptr;
    *    TimeProgressListener* time_listener = nullptr;
    *
    *   **PropensityListener**: 
    *   - Monitors rule propensities over time
    *   - Alerts if all propensities drop to zero
    *   - Tracks distribution of propensities across rules
    *   - Would attach to each geocell and be notified when propensities are calculated
    * 
    *  **RuleFireListener**:
    *    - Records which rules fire and how often
    *    - Detects cycles or patterns in rule firing
    *    - Would be notified in the reaction execution phase
    * 
    *  **TimeProgressListener**:
    *    - Monitors simulation time advancement
    *    - Reports if simulation slows down or gets stuck
    *    - Would track wall-clock vs simulation time ratio
    *
    **/

    // Finish listeners
    class PropensityListener : Listener {

    public:
        // std::vector<std::string> propensity_data;

        PropensityListener() {
            // Setting listener name
            this->setFileName("PropensityListener");
            this->createCSVFile();
            this->writeHeader({"SimStep", "SSAStep", "CellID", "Propensity"});
        };

        void onUpdate() override {
            std::cout << "PropensityListener Update\n";
        }

        void getInfo() override {
            std::cout << "PropensityListener Info\n";
        }

        void onPropensityCalculated(int sim_step, int ssa_step, std::size_t cell_id, double propensity) {

            this->writeDataRow({std::to_string(sim_step),
                                 std::to_string(ssa_step),
                                 std::to_string(cell_id),
                                 std::to_string(propensity)});

            if(propensity == 0.0) {
                zero_propensity_cells.insert(cell_id);
                // std::cout << "Warning: Cell " << cell_id << " has zero propensity!\n";

            } else {
                zero_propensity_cells.erase(cell_id);
            }

        }

        const std::set<std::size_t>& getZeroPropensityCells() const {
            return zero_propensity_cells;
        }

    private:
        std::set<std::size_t> zero_propensity_cells;
    };

    class RuleFireListener : Listener {

    public:

        RuleFireListener() {
            this->setFileName("RuleFireListener");
            this->createCSVFile();
            this->writeHeader({"SimStep", "SSAStep", "RuleName", "FireCount"});
        };

        void onUpdate() override {
            std::cout << "RuleFireListener Update\n";
        }

        void getInfo() override {
            std::cout << "RuleFireListener Info\n";
        }

        void onRuleFired(int sim_step, int ssa_step, const std::string& rule_name) {
            rule_fire_count[rule_name]++;
            // std::cout << "Rule " << rule_name << " fired. Total fired count: " << rule_fire_count[rule_name] << "\n";
            this->writeDataRow({std::to_string(sim_step),
                                 std::to_string(ssa_step),
                                 rule_name,
                                 std::to_string(rule_fire_count[rule_name])});
        }

        const std::map<std::string, std::size_t>& getRuleFireCounts() const {
            return rule_fire_count;
        }

    private:
        std::map<std::string, std::size_t> rule_fire_count;
        std::map<std::string, std::size_t> rule_changes;
    };

    class TimeProgressListener : Listener {

    public:

        TimeProgressListener() : last_sim_time(0.0), last_wall_time(std::chrono::high_resolution_clock::now()) {
            this->setFileName("TimeProgressListener");
            this->createCSVFile();
            this->writeHeader({"SimStep", "SSAStep", "SimTime", "WallTime", "Ratio"});
        };

        void onUpdate() override {
            std::cout << "TimeProgressListener Update\n";
        }

        void getInfo() override {
            std::cout << "TimeProgressListener Info\n";
        }

        void onTimeAdvanced(int sim_step, int ssa_step, double sim_time) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> wall_duration = now - last_wall_time;
            double wall_time = wall_duration.count();

            // double sim_delta = last_sim_time - sim_time;
            double sim_delta = sim_time;

            if(wall_time > 0) {
                double ratio = sim_delta / wall_time;

                this->writeDataRow({std::to_string(sim_step),
                                     std::to_string(ssa_step),
                                     std::to_string(sim_time),
                                     std::to_string(wall_time),
                                     std::to_string(ratio)});

                // std::cout << "Simulation time advanced by " << sim_delta << " in " << wall_time << " seconds. Ratio: " << ratio << "\n";

                if(ratio < threshold) {
                    // std::cout << "Warning: Simulation is slowing down! Ratio: " << ratio << "\n";
                }
            }

            last_sim_time = sim_time;
            last_wall_time = now;
        }

    private:
        double last_sim_time;
        std::chrono::high_resolution_clock::time_point last_wall_time;
        const double threshold = 0.1; // Example threshold ratio
    };

    class InvalidationValidationListener : Listener {
    public:
        InvalidationValidationListener() {
            this->setFileName("InvalidationValidationListener");
            this->createCSVFile();
            this->writeHeader({"SimStep", "SSAStep", "InvalidatedComponentsNode", "ValidatedComponentsNode",
                    "InvalidatedComponentsEdge", "ValidatedComponentsEdge"});
        };

        void onUpdate() override {
            std::cout << "InvalidationValidationListener Update\n";
        }

        void getInfo() override {
            std::cout << "InvalidationValidationListener Info\n";
        }

        void onComponentsInvalidatedValidated(int sim_step, int ssa_step,
                                              const std::vector<std::size_t>& invalidated_node,
                                              const std::vector<std::size_t>& validated_node,
                                              const std::vector<std::pair<std::size_t, std::size_t>>& invalidated_edge,
                                              const std::vector<std::pair<std::size_t, std::size_t>>& validated_edge) {

            // Convert vectors to string representations
            auto vectorToString = [](const std::vector<std::size_t>& vec) {
                std::string result = "[";
                for (size_t i = 0; i < vec.size(); ++i) {
                    result += std::to_string(vec[i]);
                    if (i < vec.size() - 1) {
                        result += ";";
                    }
                }
                result += "]";
                return result;
            };

            auto edgeVectorToString = [](const std::vector<std::pair<std::size_t, std::size_t>>& vec) {
                std::string result = "[";
                for (size_t i = 0; i < vec.size(); ++i) {
                    result += "(" + std::to_string(vec[i].first) + "-" + std::to_string(vec[i].second) + ")";
                    if (i < vec.size() - 1) {
                        result += ";";
                    }
                }
                result += "]";
                return result;
            };

            std::string invalidated_node_str = vectorToString(invalidated_node);
            std::string validated_node_str = vectorToString(validated_node);
            std::string invalidated_edge_str = edgeVectorToString(invalidated_edge);
            std::string validated_edge_str = edgeVectorToString(validated_edge);
            this->writeDataRow({std::to_string(sim_step),
                                 std::to_string(ssa_step),
                                 invalidated_node_str,
                                 validated_node_str,
                                 invalidated_edge_str,
                                 validated_edge_str});
        }

    };


    /**
     * Keeps track of all the listeners and notifies them of events
     */
    class LivelinessMonitor {
    public:

        int ssa_steps = 0;
        int simulation_steps = 0;

        LivelinessMonitor() {
            // Creating listeners
            PropensityListener* p_listener = new PropensityListener();
            RuleFireListener* rf_listener = new RuleFireListener();
            TimeProgressListener* tp_listener = new TimeProgressListener();

            InvalidationValidationListener* iv_listener = new InvalidationValidationListener();

            // Registering Listeners
            addPropensityListener(p_listener);
            addRuleFireListener(rf_listener);
            addTimeProgressListener(tp_listener);
            addInvalidationValidationListener(iv_listener);
        };

        void addInvalidationValidationListener(InvalidationValidationListener* listener) {
            invalidation_validation_listener = listener;
        }

        void addPropensityListener(PropensityListener* listener) {
            propensity_listener = listener;
        }

        void addRuleFireListener(RuleFireListener* listener) {
            rule_fire_listener = listener;
        }

        void addTimeProgressListener(TimeProgressListener* listener) {
            time_listener = listener;
        }

        void notifyPropensityCalculated(std::size_t cell_id, double propensity) {
            if(propensity_listener)
                propensity_listener->onPropensityCalculated(this->simulation_steps,
                                                            this->ssa_steps,
                                                            cell_id, propensity);
        }

        void notifyRuleFired(const std::string& rule_name) {

            if(rule_fire_listener)
                rule_fire_listener->onRuleFired(this->simulation_steps,
                                                this->ssa_steps,
                                                rule_name);

        }

        void notifyTimeAdvanced(double sim_time) {
            if(time_listener)
                time_listener->onTimeAdvanced(this->simulation_steps, this->ssa_steps, sim_time);
        }

        void notifyComponentsInvalidatedValidated(RewriteUpdates& changes,
                                                 Invalidations& invalidations) {

            auto invalidated_node = std::vector<std::size_t>(
                        changes.node_removals.begin(),
                        changes.node_removals.end()
                    );

            auto validated_node = std::vector<std::size_t>(
                        changes.node_updates.begin(),
                        changes.node_updates.end()
                    );

            auto invalidated_edge = std::vector<std::pair<std::size_t, std::size_t>>(
                        changes.edge_removals.begin(),
                        changes.edge_removals.end()
                    );

            auto validated_edge = std::vector<std::pair<std::size_t, std::size_t>>(
                        changes.edge_updates.begin(),
                        changes.edge_updates.end()
                    );

            if(invalidation_validation_listener)
                invalidation_validation_listener->onComponentsInvalidatedValidated(
                            this->simulation_steps,
                            this->ssa_steps,
                            invalidated_node,
                            validated_node,
                            invalidated_edge,
                            validated_edge
                        );

        }

        void notifySSAstep() {
            ssa_steps++;
        }

        void notifySimulationStep() {
            simulation_steps++;
        }

    private:
        PropensityListener* propensity_listener = nullptr;
        RuleFireListener* rule_fire_listener = nullptr;
        TimeProgressListener* time_listener = nullptr;
        InvalidationValidationListener* invalidation_validation_listener = nullptr;
    };

}
#endif //DGGML_LIVELINESSMONITOR_HPP
