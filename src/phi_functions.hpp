#ifndef DGGML_PHI_FUNCTIONS_HPP
#define DGGML_PHI_FUNCTIONS_HPP

#include <cmath>
#include <iostream>

template<typename RuleInstance, typename GraphType, typename CellComplexType>
std::size_t anchored_phi(RuleInstance& inst, GraphType& system_graph, CellComplexType& geoplex2D)
{
    auto& node_data = system_graph.findNode(inst.anchor)->second.getData();

    double xp = node_data.position[0];
    double yp = node_data.position[1];
    int ic, jc;

    geoplex2D.reaction_grid.locatePoint(xp, yp, ic, jc);
    auto cardinal = geoplex2D.reaction_grid.cardinalCellIndex(ic, jc);
    
    // Bounds check
    int total_cells = geoplex2D.reaction_grid.totalNumCells();
    if(cardinal < 0 || cardinal >= total_cells) {
        std::cerr << "[anchored_phi 2D] WARNING: anchor at (" << xp << ", " << yp 
                  << ") mapped to out-of-bounds cardinal " << cardinal << ". Returning geocell 0.\n";
        return 0;
    }

    auto max_cell = geoplex2D.cell_label[cardinal];

    return max_cell;
}

template<typename RuleInstance, typename GraphType, typename CellComplexType, typename ComponentType>
std::size_t min_dim_phi(RuleInstance& inst, GraphType& system_graph,
        CellComplexType& geoplex2D, ComponentType& component_matches)
{
    std::size_t min_cell;
    std::size_t min_dim;
    for(auto i = 0; i < inst.components.size(); i++)
    {
        auto& m = component_matches[inst.components[i]].match;
        for(auto j = 0; j < m.size(); j++)
        {
            auto& item = m[j];
            auto& node_data = system_graph.findNode(item)->second.getData();

            double xp = node_data.position[0];
            double yp = node_data.position[1];
            int ic, jc;
            geoplex2D.reaction_grid.locatePoint(xp, yp, ic, jc);
            auto cardinal = geoplex2D.reaction_grid.cardinalCellIndex(ic, jc);
            
            // Bounds check
            int total_cells = geoplex2D.reaction_grid.totalNumCells();
            if(cardinal < 0 || cardinal >= total_cells) {
                std::cerr << "[min_dim_phi 2D] WARNING: node " << item 
                          << " at (" << xp << ", " << yp 
                          << ") mapped to out-of-bounds cardinal " << cardinal << ". Skipping.\n";
                continue;
            }

            if(i == 0 && j == 0)
            {
                //TODO: dim label are incorrectly in reverse order in the ECC
                min_dim = geoplex2D.dim_label[cardinal];
                min_cell = geoplex2D.cell_label[cardinal];
            } else
            {
                auto temp_min_dim = geoplex2D.dim_label[cardinal];
                auto temp_min_cell = geoplex2D.cell_label[cardinal];
                //TODO: we need to fix the bug above to change the sign
                if(temp_min_dim > min_dim)
                {
                    min_dim = temp_min_dim;
                    min_cell = temp_min_cell;
                }
            }
        }
    }

    return min_cell;
}


namespace DGGML3D {
/**
 * @brief Min Dim Phi: Finds the lowest-dimension feature occupied by any node in the rule instance.
 * Useful for reactions that must occur on boundaries (e.g., surface-bound reactions).
 */
template<typename RuleInstance, typename GraphType, typename CellComplexType, typename ComponentType>
std::size_t min_dim_phi(RuleInstance& inst, GraphType& system_graph,
        CellComplexType& geoplex3D, ComponentType& component_matches)
    {
        std::size_t min_cell_key;
        int min_dim = 999; // Initialize with high value to find minimum

        // Iterate through all components of the rule instance
        for(auto i = 0; i < inst.components.size(); i++)
        {
            auto& m = component_matches[inst.components[i]].match;
            for(auto j = 0; j < m.size(); j++)
            {
                auto& node_id = m[j];
                auto& node_data = system_graph.findNode(node_id)->second.getData();

                double xp = node_data.position[0];
                double yp = node_data.position[1];
                double zp = node_data.position[2]; // Added Z
                int ic, jc, kc; // Added kc
                
                geoplex3D.reaction_grid.locatePoint(xp, yp, zp, ic, jc, kc);
                auto cardinal = geoplex3D.reaction_grid.cardinalCellIndex(ic, jc, kc);
                
                // Bounds check: cardinal must be within the reaction grid
                int total_cells = geoplex3D.reaction_grid.totalNumCells();
                if(cardinal < 0 || cardinal >= total_cells) {
                    std::cerr << "[min_dim_phi] WARNING: node " << node_id 
                              << " at (" << xp << ", " << yp << ", " << zp 
                              << ") mapped to out-of-bounds cardinal " << cardinal
                              << " (max=" << total_cells - 1 << "), ic=" << ic 
                              << " jc=" << jc << " kc=" << kc << ". Skipping.\n";
                    continue;
                }

                // In the 3D Expanded Complex:
                // dim_label usually: 0=3D Volume, 1=2D Face, 2=1D Edge, 3=0D Vertex
                int current_dim = geoplex3D.dim_label[cardinal];
                std::size_t current_cell_key = geoplex3D.cell_label[cardinal];

                // We look for the "lowest" dimension (e.g., if a node is on an Edge (2), 
                // it overrides a node in a Volume (0)).
                if((i == 0 && j == 0) || current_dim > min_dim)
                {
                    min_dim = current_dim;
                    min_cell_key = current_cell_key;
                }
            }
        }
        return min_cell_key;
    }

/**
 * @brief Anchored Phi: Localizes a rule instance based on the anchor node's 3D position.
 */
template<typename RuleInstance, typename GraphType, typename CellComplexType>
std::size_t anchored_phi(RuleInstance& inst, GraphType& system_graph, CellComplexType& geoplex3D)
{
    auto& node_data = system_graph.findNode(inst.anchor)->second.getData();

    double xp = node_data.position[0];
    double yp = node_data.position[1];
    double zp = node_data.position[2]; // Added Z
    int ic, jc, kc; // Added kc

    // Perform 3D point location
    geoplex3D.reaction_grid.locatePoint(xp, yp, zp, ic, jc, kc);
    auto cardinal = geoplex3D.reaction_grid.cardinalCellIndex(ic, jc, kc);
    
    // Bounds check
    int total_cells = geoplex3D.reaction_grid.totalNumCells();
    if(cardinal < 0 || cardinal >= total_cells) {
        std::cerr << "[anchored_phi] WARNING: anchor node at (" << xp << ", " << yp << ", " << zp 
                  << ") mapped to out-of-bounds cardinal " << cardinal
                  << " (max=" << total_cells - 1 << "), ic=" << ic 
                  << " jc=" << jc << " kc=" << kc << ". Returning geocell 0.\n";
        return 0;
    }

    // Return the label (Key) of the topological feature (Volume, Face, etc.)
    auto geocell_key = geoplex3D.cell_label[cardinal];

    return geocell_key;
}



}


#endif //DGGML_PHI_FUNCTIONS_HPP
