//
// Created by Eric Medwedeff on 7/20/23.
//

#ifndef DGGML_SIMULATORINTERFACE_H
#define DGGML_SIMULATORINTERFACE_H

#include <memory>
#include <type_traits>
#include "SimulationRunner.h"

namespace DGGML {

    // DomainParallel selects the runner at compile time: false -> SimulationRunner (serial),
    // true -> SimulationRunnerParallel (OpenMP geocell domain decomposition).
    template<typename ModelType, int Dim = 3, bool DomainParallel = false>
    class SimulatorInterface {
    public:
        using runner_type = std::conditional_t<DomainParallel,
                                               SimulationRunnerParallel<ModelType, Dim>,
                                               SimulationRunner<ModelType, Dim>>;

        bool is_3D = false;

        void setModel(const ModelType& model)
        {
                simRunner = std::make_unique<runner_type>(model);
        }

        void set3D(bool val) {
            is_3D = val;
	    simRunner->is_3D = val;
        }

        void simulate()
        {
            if(simRunner)
            {
                simRunner->initialize();
                simRunner->run();
            }
            else
            {
                std::cout << "Error: no model set!";
            }
        }
    private:
        std::unique_ptr<runner_type> simRunner;
    };

} // DGGML

#endif //DGGML_SIMULATORINTERFACE_H
