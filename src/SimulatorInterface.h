//
// Created by Eric Medwedeff on 7/20/23.
//

#ifndef DGGML_SIMULATORINTERFACE_H
#define DGGML_SIMULATORINTERFACE_H

#include "SimulationRunner.h"

namespace DGGML {

    template<typename ModelType, int Dim = 3>
    class SimulatorInterface {
    public:

        bool is_3D = false;

        void setModel(const ModelType& model)
        {
                simRunner = std::make_unique<SimulationRunner<ModelType, Dim>>(model);
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
        std::unique_ptr<SimulationRunner<ModelType, Dim>> simRunner;
    };

} // DGGML

#endif //DGGML_SIMULATORINTERFACE_H
