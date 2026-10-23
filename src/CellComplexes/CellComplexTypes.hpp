#ifndef DGGML_CELL_COMPLEX_TYPES_HPP
#define DGGML_CELL_COMPLEX_TYPES_HPP

#include "YAGL_Graph.hpp"

namespace DGGML
{
    // --- 2D Definitions (Kept for reference, based on original file) ---
    enum corner_types_2d
    {
        lower_left,
        lower_right,
        upper_left,
        upper_right
    };

    struct cell_complex_2D_node_t
    {
        std::size_t type; 
        double position[3]; 
        double corners[4][2]; // 4 corners, 2 components
        bool ghosted;
        bool interior;
    };
    
    using cell_complex_2D_key_t = std::size_t;
    using CplexGraph2D_t = YAGL::Graph<cell_complex_2D_key_t, cell_complex_2D_node_t>;    
    
    // --- 3D Definitions ---
    enum corner_types_3d
    {
        lower_left_back,
        lower_right_back,
        upper_left_back,
        upper_right_back,
        lower_left_front,
        lower_right_front,
        upper_left_front,
        upper_right_front
    };

    struct cell_complex_3D_node_t
    {
        std::size_t type; // 0=3D, 1=2D, 2=1D, 3=0D
        double position[3]; // Center (x, y, z)
        double corners[8][3]; // 8 corners, 3 components for a 3D cell
        bool ghosted;
        bool interior;
    };
    
    using cell_complex_3D_key_t = std::size_t;
    using CplexGraph3D_t = YAGL::Graph<cell_complex_3D_key_t, cell_complex_3D_node_t>;    

}// end namespace DGGML

#endif
