#ifndef DGGML_CARTESIAN_COMPLEX_3D_HPP
#define DGGML_CARTESIAN_COMPLEX_3D_HPP

#include "CellComplexTypes.hpp"
// Assuming a corresponding 3D grid class exists
#include "CartesianGrid3D.hpp" 

namespace DGGML
{
    // Assuming CplexGraph3D_t is defined in CellComplexTypes.hpp
    template <typename GraphType = CplexGraph3D_t>
    class CartesianComplex3D
    {
        public:
            using graph_type = GraphType;
            using node_type = typename graph_type::node_type;
            using edge_type = typename graph_type::edge_type;
            using data_type = typename graph_type::data_type;
            
            CartesianComplex3D() = default;

            CartesianComplex3D(std::size_t n, std::size_t m, std::size_t l, double d_x, double d_y, double d_z, bool g_c = false)
            {
                init(n, m, l, d_x, d_y, d_z, g_c);
            }
            
            void init(std::size_t n, std::size_t m, std::size_t l, double d_x, double d_y, double d_z, bool g_c = false)
            {
                // Points per cell
                ppc = 2; // Fixed for 3D lattice
                 
                ghost_cells = g_c;
                if(g_c)
                {
                    n = n+2; // pad around boundary in x
                    m = m+2; // pad around boundary in y
                    l = l+2; // pad around boundary in z
                } 

                nx = n * ppc; 
                ny = m * ppc;
                nz = l * ppc;

                dx = d_x; 
                dy = d_y;
                dz = d_z;

                // Assuming CartesianGrid3D takes (min_x, min_y, min_z, max_x, max_y, max_z, nx, ny, nz)
                fine_grid.init(0.0, 0.0, 0.0, n*dx, m*dy, l*dz, nx, ny, nz);
                coarse_grid.init(0.0, 0.0, 0.0, n*dx, m*dy, l*dz, n, m, l);

                nx = fine_grid._px;
                ny = fine_grid._py;
                nz = fine_grid._pz;

                // Boundary box calculations
                if(ghost_cells) 
                {
                    min_x = coarse_grid._min_x + coarse_grid._dx;
                    min_y = coarse_grid._min_y + coarse_grid._dy;
                    min_z = coarse_grid._min_z + coarse_grid._dz;
                    max_x = coarse_grid._max_x - coarse_grid._dx;
                    max_y = coarse_grid._max_y - coarse_grid._dy;
                    max_z = coarse_grid._max_z - coarse_grid._dz;
                } else
                {
                    min_x = coarse_grid._min_x;
                    min_y = coarse_grid._min_y;
                    min_z = coarse_grid._min_z;
                    max_x = coarse_grid._max_x;
                    max_y = coarse_grid._max_y;
                    max_z = coarse_grid._max_z;
                }

                // Cell count initialization (now includes 3D cells)
                num_interior_0D_cells = 0;
                num_exterior_0D_cells = 0;
                num_interior_1D_cells = 0;
                num_exterior_1D_cells = 0;
                num_interior_2D_cells = 0;
                num_exterior_2D_cells = 0;
                num_interior_3D_cells = 0;
                num_exterior_3D_cells = 0;

                build();
            }

            std::size_t get3dInteriorCellCount() { return num_interior_3D_cells; }
            std::size_t get3dExteriorCellCount() { return num_exterior_3D_cells; }
            std::size_t get3dTotalCellCount() { return num_interior_3D_cells + num_exterior_3D_cells; }

            // Update Total Cell Counts
            std::size_t getTotalInteriorCellCount()
            {
                return get0dInteriorCellCount() + get1dInteriorCellCount() + get2dInteriorCellCount() + get3dInteriorCellCount();
            }

            std::size_t getTotalExteriorCellCount()
            {
                return get0dExteriorCellCount() + get1dExteriorCellCount() + get2dExteriorCellCount() + get3dExteriorCellCount();
            }

            std::size_t get0dInteriorCellCount()
            {
                return num_interior_0D_cells;
            }

            std::size_t get0dExteriorCellCount()
            {
                return num_exterior_0D_cells;
            }
            
            std::size_t get0dTotalCellCount()
            {
                return num_interior_0D_cells + num_exterior_0D_cells;
            }

            std::size_t get1dInteriorCellCount()
            {
                return num_interior_1D_cells;
            }
            
            std::size_t get1dExteriorCellCount()
            {
                return num_exterior_1D_cells;
            }
            
            std::size_t get1dTotalCellCount()
            {
                return num_interior_1D_cells + num_exterior_1D_cells;
            }

            std::size_t get2dInteriorCellCount()
            {
                return num_interior_2D_cells;
            }

            std::size_t get2dExteriorCellCount()
            {
                return num_exterior_2D_cells;
            }
            
            std::size_t get2dTotalCellCount()
            {
                return num_interior_2D_cells + num_exterior_2D_cells;
            }
             
            std::size_t getTotalCellCount() 
            {
                return get0dTotalCellCount() + get1dTotalCellCount() + get2dTotalCellCount();
            }
            
            //TODO: generate lists of interior cells 
            friend std::ostream& operator<<(std::ostream& os, CartesianComplex3D& cplex)
            {
                os << "\nCoarse Grid: \n" << cplex.coarse_grid;
                os << "\nFine Grid: \n" << cplex.fine_grid;
                os << cplex.graph;
                
                return os;
            }

        public:


            void build()
            {
                // Iterate over all points in the fine grid, including ghost layers (from 0 to px, py, pz - 1)
                for(int k = 0; k < fine_grid._pz; k++)
                {
                    for(int j = 0; j < fine_grid._py; j++)
                    {
                        for(int i = 0; i < fine_grid._px; i++)
                        {
                            // Determine cell type based on parity: 0=odd, 1=even
                            int p_i = i % ppc; // 1 if odd, 0 if even
                            int p_j = j % ppc;
                            int p_k = k % ppc;

                            // Calculate cell dimension type (0:3D, 1:2D, 2:1D, 3:0D)
                            std::size_t type = p_i + p_j + p_k;

                            // Only proceed if the cell is a center of a dimension (i.e., not a boundary or vertex)
                            // A center of dimension 'd' has (3-d) odd indices.
                            // Example: 3D center (d=0) is (o, o, o) -> type = 3
                            // Example: 2D face (d=1) is (e, o, o) -> type = 2

                            // 0D cell (Vertex/Node): i, j, k all even (type = 0). It is a terminal node, no k->k-1 connections.
                            if (type == 0) continue;

                            ///
                            // Inside CartesianComplex3D::build()
                            // ...

                            // Check for interior cell centers (i, j, k are all odd, type = 3)
                            if (type == 3) // 3D Cell (Type 0) at (o, o, o)
                            {
                                // Connect 3D (i, j, k) -> ALL 6 adjacent 2D Faces (Type 1)
                                
                                // X-Faces (i+/-1, j, k)
                                create_edge(i, j, k, 0, i+1, j, k, 1); // +X Face
                                create_edge(i, j, k, 0, i-1, j, k, 1); // -X Face
                                
                                // Y-Faces (i, j+/-1, k)
                                create_edge(i, j, k, 0, i, j+1, k, 1); // +Y Face
                                create_edge(i, j, k, 0, i, j-1, k, 1); // -Y Face
                                
                                // Z-Faces (i, j, k+/-1)
                                create_edge(i, j, k, 0, i, j, k+1, 1); // +Z Face
                                create_edge(i, j, k, 0, i, j, k-1, 1); // -Z Face
                            }

                            // Check for 2D cell centers (one index even, two odd, type = 2)
                            else if (type == 2) // 2D Face (Type 1)
                            {
                                // X-normal Face (i even, j odd, k odd): Connect to ALL 4 adjacent 1D Edges (Type 2)
                                if (p_i == 0 && p_j == 1 && p_k == 1)
                                {
                                    // Y-direction Edges (i, j+/-1, k)
                                    create_edge(i, j, k, 1, i, j+1, k, 2); // +Y Edge
                                    create_edge(i, j, k, 1, i, j-1, k, 2); // -Y Edge

                                    // Z-direction Edges (i, j, k+/-1)
                                    create_edge(i, j, k, 1, i, j, k+1, 2); // +Z Edge
                                    create_edge(i, j, k, 1, i, j, k-1, 2); // -Z Edge
                                }
                                // Y-normal Face (j even, i odd, k odd): Connect to ALL 4 adjacent 1D Edges
                                else if (p_i == 1 && p_j == 0 && p_k == 1)
                                {
                                    // X-direction Edges (i+/-1, j, k)
                                    create_edge(i, j, k, 1, i+1, j, k, 2); // +X Edge
                                    create_edge(i, j, k, 1, i-1, j, k, 2); // -X Edge

                                    // Z-direction Edges (i, j, k+/-1)
                                    create_edge(i, j, k, 1, i, j, k+1, 2); // +Z Edge
                                    create_edge(i, j, k, 1, i, j, k-1, 2); // -Z Edge
                                }
                                // Z-normal Face (k even, i odd, j odd): Connect to ALL 4 adjacent 1D Edges
                                else if (p_i == 1 && p_j == 1 && p_k == 0)
                                {
                                    // X-direction Edges (i+/-1, j, k)
                                    create_edge(i, j, k, 1, i+1, j, k, 2); // +X Edge
                                    create_edge(i, j, k, 1, i-1, j, k, 2); // -X Edge

                                    // Y-direction Edges (i, j+/-1, k)
                                    create_edge(i, j, k, 1, i, j+1, k, 2); // +Y Edge
                                    create_edge(i, j, k, 1, i, j-1, k, 2); // -Y Edge
                                }
                            }

                            // Check for 1D cell centers (two indices even, one odd, type = 1)
                            else if (type == 1) // 1D Edge (Type 2)
                            {
                                // X-parallel Edge (i odd, j even, k even): Connect to ALL 2 adjacent 0D Vertices (Type 3)
                                if (p_i == 1 && p_j == 0 && p_k == 0)
                                {
                                    create_edge(i, j, k, 2, i+1, j, k, 3); // +X Vertex
                                    create_edge(i, j, k, 2, i-1, j, k, 3); // -X Vertex
                                }
                                // Y-parallel Edge (j odd, i even, k even): Connect to ALL 2 adjacent 0D Vertices
                                else if (p_i == 0 && p_j == 1 && p_k == 0)
                                {
                                    create_edge(i, j, k, 2, i, j+1, k, 3); // +Y Vertex
                                    create_edge(i, j, k, 2, i, j-1, k, 3); // -Y Vertex
                                }
                                // Z-parallel Edge (k odd, i even, j even): Connect to ALL 2 adjacent 0D Vertices
                                else if (p_i == 0 && p_j == 0 && p_k == 1)
                                {
                                    create_edge(i, j, k, 2, i, j, k+1, 3); // +Z Vertex
                                    create_edge(i, j, k, 2, i, j, k-1, 3); // -Z Vertex
                                }
                            }

// ... rest of build() ...

                            ///


                            /*
                            // Check for interior cell centers (i, j, k are all odd, type = 3)
                            if (type == 3) // 3D Cell (Type 0) at (o, o, o)
                            {
                                // Connect 3D (i, j, k) -> 2D Face (i +/- 1, j, k)
                                // Only create edges in the positive direction to avoid duplicates
                                create_edge(i, j, k, 0, i+1, j, k, 1); // +X Face
                                create_edge(i, j, k, 0, i, j+1, k, 1); // +Y Face
                                create_edge(i, j, k, 0, i, j, k+1, 1); // +Z Face
                            }

                            // Check for 2D cell centers (one index even, two odd, type = 2)
                            else if (type == 2) // 2D Face (Type 1)
                            {
                                // X-normal Face (i even, j odd, k odd): Connect to 1D Edges in Y and Z directions
                                if (p_i == 0 && p_j == 1 && p_k == 1)
                                {
                                    // Connect 2D (i, j, k) -> 1D Edge in Y direction
                                    create_edge(i, j, k, 1, i, j+1, k, 2);

                                    // Connect 2D (i, j, k) -> 1D Edge in Z direction
                                    create_edge(i, j, k, 1, i, j, k+1, 2);
                                }
                                // Y-normal Face (j even, i odd, k odd): Connect to 1D Edges in X and Z directions
                                else if (p_i == 1 && p_j == 0 && p_k == 1)
                                {
                                    // Connect 2D (i, j, k) -> 1D Edge in X direction
                                    create_edge(i, j, k, 1, i+1, j, k, 2);
                                    // Connect 2D (i, j, k) -> 1D Edge in Z direction
                                    create_edge(i, j, k, 1, i, j, k+1, 2);
                                }
                                // Z-normal Face (k even, i odd, j odd): Connect to 1D Edges in X and Y directions
                                else if (p_i == 1 && p_j == 1 && p_k == 0)
                                {
                                    // Connect 2D (i, j, k) -> 1D Edge in X direction
                                    create_edge(i, j, k, 1, i+1, j, k, 2);
                                    // Connect 2D (i, j, k) -> 1D Edge in Y direction
                                    create_edge(i, j, k, 1, i, j+1, k, 2);
                                }
                            }

                            // Check for 1D cell centers (two indices even, one odd, type = 1)
                            else if (type == 1) // 1D Edge (Type 2)
                            {
                                // X-parallel Edge (i odd, j even, k even): Connect to 0D Vertices in X direction
                                if (p_i == 1 && p_j == 0 && p_k == 0)
                                {
                                    create_edge(i, j, k, 2, i+1, j, k, 3); // +X Vertex
                                }
                                // Y-parallel Edge (j odd, i even, k even): Connect to 0D Vertices in Y direction
                                else if (p_i == 0 && p_j == 1 && p_k == 0)
                                {
                                    create_edge(i, j, k, 2, i, j+1, k, 3); // +Y Vertex
                                }
                                // Z-parallel Edge (k odd, i even, j even): Connect to 0D Vertices in Z direction
                                else if (p_i == 0 && p_j == 0 && p_k == 1)
                                {
                                    create_edge(i, j, k, 2, i, j, k+1, 3); // +Z Vertex
                                }
                            }
                            */


                        }
                    }
                }

                // Count interior/exterior cells (existing counting logic preserved)
                num_interior_0D_cells = 0;
                num_exterior_0D_cells = 0;
                num_interior_1D_cells = 0;
                num_exterior_1D_cells = 0;
                num_interior_2D_cells = 0;
                num_exterior_2D_cells = 0;
                num_interior_3D_cells = 0;
                num_exterior_3D_cells = 0;

                for(auto iter = graph.node_list_begin(); iter != graph.node_list_end(); iter++)
                {
                    auto& data = iter->second.getData();

                    if (data.ghosted) {
                        data.interior = false;
                    } else {
                        data.interior = true;
                    }

                    if(data.type == 0) (data.interior ? num_interior_3D_cells : num_exterior_3D_cells)++;
                    else if(data.type == 1) (data.interior ? num_interior_2D_cells : num_exterior_2D_cells)++;
                    else if(data.type == 2) (data.interior ? num_interior_1D_cells : num_exterior_1D_cells)++;
                    else if(data.type == 3) (data.interior ? num_interior_0D_cells : num_exterior_0D_cells)++;
                }
            }
            
            /*
            void build()
            {
                // Iterate over the center of the coarse 3D cells on the fine grid
                for(int k = 1; k < fine_grid._pz; k += 2)
                {
                    for(int j = 1; j < fine_grid._py; j += 2)
                    {
                        for(int i = 1; i < fine_grid._px; i += 2)
                        {
                            // A 3D cell (i, j, k, type=0) is connected to the 6 adjacent 2D cells (faces, type=1)
                            // 3D (cell center) -> 2D (face centers)
                            create_edge(i, j, k, 0, i+1, j, k, 1); // +x face
                            create_edge(i, j, k, 0, i-1, j, k, 1); // -x face
                            create_edge(i, j, k, 0, i, j+1, k, 1); // +y face
                            create_edge(i, j, k, 0, i, j-1, k, 1); // -y face
                            create_edge(i, j, k, 0, i, j, k+1, 1); // +z face
                            create_edge(i, j, k, 0, i, j, k-1, 1); // -z face

                            // A 2D cell (face center, type=1) is connected to the 4 adjacent 1D cells (edges, type=2)
                            
                            // 2D Face at (i+1, j, k) -> 4 Edges
                            create_edge(i+1, j, k, 1, i+1, j+1, k, 2); // +y edge
                            create_edge(i+1, j, k, 1, i+1, j-1, k, 2); // -y edge
                            create_edge(i+1, j, k, 1, i+1, j, k+1, 2); // +z edge
                            create_edge(i+1, j, k, 1, i+1, j, k-1, 2); // -z edge

                            // 2D Face at (i, j+1, k) -> 4 Edges
                            create_edge(i, j+1, k, 1, i+1, j+1, k, 2); // +x edge
                            create_edge(i, j+1, k, 1, i-1, j+1, k, 2); // -x edge
                            create_edge(i, j+1, k, 1, i, j+1, k+1, 2); // +z edge
                            create_edge(i, j+1, k, 1, i, j+1, k-1, 2); // -z edge

                            // 2D Face at (i, j, k+1) -> 4 Edges
                            create_edge(i, j, k+1, 1, i+1, j, k+1, 2); // +x edge
                            create_edge(i, j, k+1, 1, i-1, j, k+1, 2); // -x edge
                            create_edge(i, j, k+1, 1, i, j+1, k+1, 2); // +y edge
                            create_edge(i, j, k+1, 1, i, j-1, k+1, 2); // -y edge
                            
                            // A 1D cell (edge center, type=2) is connected to the 2 adjacent 0D cells (vertices, type=3)
                            
                            // 1D Edge at (i+1, j+1, k) -> 2 Vertices
                            create_edge(i+1, j+1, k, 2, i+1, j+1, k+1, 3); // +z vertex
                            create_edge(i+1, j+1, k, 2, i+1, j+1, k-1, 3); // -z vertex

                            // 1D Edge at (i+1, j, k+1) -> 2 Vertices
                            create_edge(i+1, j, k+1, 2, i+1, j+1, k+1, 3); // +y vertex
                            create_edge(i+1, j, k+1, 2, i+1, j-1, k+1, 3); // -y vertex
                            
                            // 1D Edge at (i, j+1, k+1) -> 2 Vertices
                            create_edge(i, j+1, k+1, 2, i+1, j+1, k+1, 3); // +x vertex
                            create_edge(i, j+1, k+1, 2, i-1, j+1, k+1, 3); // -x vertex

                            // NOTE: The above loops only handle the positive half-space connections from the central 3D cell (i,j,k).
                            // The `create_edge` function will automatically handle the corresponding negative connections
                            // for *other* 3D cells, as it creates an undirected edge.
                        }
                    }
                }

                // Cell Count Computation
                for(auto iter = graph.node_list_begin(); iter != graph.node_list_end(); iter++)
                {
                    auto& data = iter->second.getData();
                    bool interior = true; // Assume interior until proven otherwise
                
                    // Ghost Cell Check
                    if (ghost_cells && data.ghosted)
                    {
                        interior = false;
                    } 
                    
                    data.interior = interior;

                    if(data.type == 0) // 3D cell
                    {
                        if(interior) num_interior_3D_cells++; else num_exterior_3D_cells++;
                    }
                    else if(data.type == 1) // 2D cell (Face)
                    {
                        // In non-ghosted, an interior face is connected to 2 (3D) cells. Exterior is connected to 1.
                        if (!ghost_cells)
                        {
                            int num_3D_nbrs = 0;
                            for(auto j = graph.out_neighbors_begin(iter->second); j != graph.out_neighbors_end(iter->second); j++)
                            {
                                if(graph.findNode(*j)->second.getData().type == 0) num_3D_nbrs++;
                            }
                            data.interior = (num_3D_nbrs >= 2);
                            if(data.interior) num_interior_2D_cells++; else num_exterior_2D_cells++;
                        } else {
                            if(data.interior) num_interior_2D_cells++; else num_exterior_2D_cells++;
                        }
                    }
                    else if(data.type == 2) // 1D cell (Edge)
                    {
                        // Similar logic for 1D cells (Edges): connected to 4 2D cells (Faces) for interior.
                        if (!ghost_cells)
                        {
                            int num_2D_nbrs = 0;
                            for(auto j = graph.out_neighbors_begin(iter->second); j != graph.out_neighbors_end(iter->second); j++)
                            {
                                if(graph.findNode(*j)->second.getData().type == 1) num_2D_nbrs++;
                            }
                            data.interior = (num_2D_nbrs >= 2);
                            if(data.interior) num_interior_1D_cells++; else num_exterior_1D_cells++;
                        } else {
                            if(data.interior) num_interior_1D_cells++; else num_exterior_1D_cells++;
                        }
                    }
                    else if(data.type == 3) // 0D cell (Vertex)
                    {
                        // Similar logic for 0D cells (Vertices): connected to 6 1D cells (Edges) for interior.
                        if (!ghost_cells)
                        {
                            int num_1D_nbrs = 0;
                            for(auto j = graph.out_neighbors_begin(iter->second); j != graph.out_neighbors_end(iter->second); j++)
                            {
                                if(graph.findNode(*j)->second.getData().type == 2) num_1D_nbrs++;
                            }
                            data.interior = (num_1D_nbrs >= 2);
                            if(data.interior) num_interior_0D_cells++; else num_exterior_0D_cells++;
                        } else {
                            if(data.interior) num_interior_0D_cells++; else num_exterior_0D_cells++;
                        }
                    }
                }
            }
            */
            

            void create_edge(int i_a, int j_a, int k_a, std::size_t a_t, int i_b, int j_b, int k_b, std::size_t b_t)
            {
                // a --> Represents cell at (i_a, j_a, k_a) of type a_t
                // b --> represents cell at (i_b, j_b, k_b) of type b_t
                // Use 3D cardinal index
                std::size_t a = fine_grid.cardinalLatticeIndex(i_a, j_a, k_a);
                std::size_t b = fine_grid.cardinalLatticeIndex(i_b, j_b, k_b);

                double px_a, py_a, pz_a, px_b, py_b, pz_b;
                
                // Convert the cardinal indices to fine_grid coordinates
                fine_grid.cardinalLatticeToPoint(px_a, py_a, pz_a, a);
                fine_grid.cardinalLatticeToPoint(px_b, py_b, pz_b, b);
                
                // Ghosted check logic (similar to 2D but includes z-dimension)
                bool ghosted_a = false;
                bool interior_a = true;
                if(ghost_cells)
                {
                    auto w = ppc + 1;
                    if((i_a < w || i_a >= nx - w) || 
                       (j_a < w || j_a >= ny - w) ||
                       (k_a < w || k_a >= nz - w))
                    {
                        ghosted_a = true;
                        interior_a = false;
                    }
                }
                // Within CartesianComplex3D::create_edge

                // ... existing code ...
                // Handle 3D Cell (Type 0) creation and corners
                if(a_t == 0)
                {
                    // --- CORRECTED CORNER CALCULATION LOGIC for both Node A and Node B ---

                    // Use local variables to represent the current node's indices
                    int i_curr = i_a; // Use i_b for Node B
                    int j_curr = j_a; // Use j_b for Node B
                    int k_curr = k_a; // Use k_b for Node B
                    double c[8][3];   // Use c_b[8][3] for Node B

                    // Z = k-1 (Bottom/Back Plane)
                    // 0: (i-1, j-1, k-1) - LL-B
                    fine_grid.cardinalLatticeToPoint(c[0][0], c[0][1], c[0][2],
                        fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr - 1, k_curr - 1));

                    // 1: (i+1, j-1, k-1) - LR-B
                    fine_grid.cardinalLatticeToPoint(c[1][0], c[1][1], c[1][2],
                        fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr - 1, k_curr - 1));

                    // 2: (i+1, j+1, k-1) -> This should be point 3 in the connectivity list
                    fine_grid.cardinalLatticeToPoint(c[3][0], c[3][1], c[3][2],
                        fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr + 1, k_curr - 1));

                    // 3: (i-1, j+1, k-1) -> This should be point 2 in the connectivity list
                    fine_grid.cardinalLatticeToPoint(c[2][0], c[2][1], c[2][2],
                        fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr + 1, k_curr - 1));


                    // Z = k+1 (Top/Front Plane)
                    // 4: (i-1, j-1, k+1) - LL-F
                    fine_grid.cardinalLatticeToPoint(c[4][0], c[4][1], c[4][2],
                        fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr - 1, k_curr + 1));

                    // 5: (i+1, j-1, k+1) - LR-F
                    fine_grid.cardinalLatticeToPoint(c[5][0], c[5][1], c[5][2],
                        fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr - 1, k_curr + 1));

                    // 6: (i+1, j+1, k+1) -> This should be point 7 in the connectivity list
                    fine_grid.cardinalLatticeToPoint(c[7][0], c[7][1], c[7][2],
                        fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr + 1, k_curr + 1));

                    // 7: (i-1, j+1, k+1) -> This should be point 6 in the connectivity list
                    fine_grid.cardinalLatticeToPoint(c[6][0], c[6][1], c[6][2],
                        fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr + 1, k_curr + 1));
                    
                    DGGML::cell_complex_3D_node_t node_data_a =
                        {a_t, {px_a, py_a, pz_a},
                         {{c[0][0], c[0][1], c[0][2]},
                             {c[1][0], c[1][1], c[1][2]},
                             {c[2][0], c[2][1], c[2][2]},
                             {c[3][0], c[3][1], c[3][2]},
                             {c[4][0], c[4][1], c[4][2]},
                             {c[5][0], c[5][1], c[5][2]},
                             {c[6][0], c[6][1], c[6][2]},
                             {c[7][0], c[7][1], c[7][2]}
                             },
                         ghosted_a, interior_a};

                    graph.addNode({a, node_data_a});

                    // graph.addNode({a, {a_t, {px_a, py_a, pz_a},
                        // {c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]}, ghosted_a, interior_a}});
                }
                // ... The rest of the create_edge function continues ...
                
                // NOTE: In 3D, a 3D cell (type=0) will have 8 corners.
                // For simplicity in this template, we'll only expand corners for 3D cells (type=0).
                /*
                if(a_t == 0) // 3D cell
                {
                    // Corner calculation here would be complex (8 points), but the structure is the same:
                    // The cell data needs to store the coordinates of its 8 corners.
                    // For simplicity, we use the 4x2D array structure from the 2D code, which should be updated for a 3D complex.
                    // For now, let's keep it simple or assume a 3D-specific data structure is used.
                    
                    std::cout << "Adding corner: " << "(" << px_a << ", " << py_a << ", " << pz_a << ") of type " << a_t << "\n";

                    // Simplified corner handling (should be 8 points in a 3D struct)
                    graph.addNode({a, {a_t, {px_a, py_a, pz_a}, 
                        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, ghosted_a, interior_a}}); 
                } 
                */
                else // 0D, 1D, 2D cells (centers are the only point of interest)
                {
                    std::cout << "Adding node for cell a at (" << px_a << ", " << py_a << ", " << pz_a << ") of type " << a_t << "\n";

                    graph.addNode({a, {a_t, {px_a, py_a, pz_a}, 
                        {{px_a, py_a, pz_a}, {px_a, py_a, pz_a}, {px_a, py_a, pz_a}, {px_a, py_a, pz_a}}, ghosted_a, interior_a}});

                // Get graph node count for debugging
                std::cout << "Graph now has " << graph.numNodes() << " nodes after adding cell a.\n";
                }

                // Repeat for cell B
                bool ghosted_b = false;
                bool interior_b = true;
                if(ghost_cells)
                {
                    auto w = ppc + 1;
                    if((i_b < w || i_b >= nx - w) || 
                       (j_b < w || j_b >= ny - w) ||
                       (k_b < w || k_b >= nz - w))
                    {
                        ghosted_b = true;
                        interior_b = false;
                    }
                }

                if(b_t == 0) 
                {

                // Use local variables to represent the current node's indices
                int i_curr = i_b; // Use i_b for Node B
                int j_curr = j_b; // Use j_b for Node B
                int k_curr = k_b; // Use k_b for Node B
                double c[8][3];   // Use c_b[8][3] for Node B

                // Z = k-1 (Bottom/Back Plane)
                // 0: (i-1, j-1, k-1) - LL-B
                fine_grid.cardinalLatticeToPoint(c[0][0], c[0][1], c[0][2], 
                    fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr - 1, k_curr - 1)); 

                // 1: (i+1, j-1, k-1) - LR-B
                fine_grid.cardinalLatticeToPoint(c[1][0], c[1][1], c[1][2], 
                    fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr - 1, k_curr - 1)); 

                // 2: (i+1, j+1, k-1) -> This should be point 3 in the connectivity list
                fine_grid.cardinalLatticeToPoint(c[3][0], c[3][1], c[3][2], 
                    fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr + 1, k_curr - 1)); 

                // 3: (i-1, j+1, k-1) -> This should be point 2 in the connectivity list
                fine_grid.cardinalLatticeToPoint(c[2][0], c[2][1], c[2][2], 
                    fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr + 1, k_curr - 1)); 


                // Z = k+1 (Top/Front Plane)
                // 4: (i-1, j-1, k+1) - LL-F
                fine_grid.cardinalLatticeToPoint(c[4][0], c[4][1], c[4][2], 
                    fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr - 1, k_curr + 1)); 

                // 5: (i+1, j-1, k+1) - LR-F
                fine_grid.cardinalLatticeToPoint(c[5][0], c[5][1], c[5][2], 
                    fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr - 1, k_curr + 1)); 

                // 6: (i+1, j+1, k+1) -> This should be point 7 in the connectivity list
                fine_grid.cardinalLatticeToPoint(c[7][0], c[7][1], c[7][2], 
                    fine_grid.cardinalLatticeIndex(i_curr + 1, j_curr + 1, k_curr + 1)); 

                // 7: (i-1, j+1, k+1) -> This should be point 6 in the connectivity list
                fine_grid.cardinalLatticeToPoint(c[6][0], c[6][1], c[6][2], 
                    fine_grid.cardinalLatticeIndex(i_curr - 1, j_curr + 1, k_curr + 1));                    
                    // auto c_b = c;
                                    // Explicitly construct node data with correct bracing
                    DGGML::cell_complex_3D_node_t node_data_b = 
                        {b_t, {px_b, py_b, pz_b},
                         {
                            {c[0][0], c[0][1], c[0][2]},
                            {c[1][0], c[1][1], c[1][2]},
                            {c[2][0], c[2][1], c[2][2]},
                            {c[3][0], c[3][1], c[3][2]},
                            {c[4][0], c[4][1], c[4][2]},
                            {c[5][0], c[5][1], c[5][2]},
                            {c[6][0], c[6][1], c[6][2]},
                            {c[7][0], c[7][1], c[7][2]}
                         },
                          ghosted_b, interior_b};

                    graph.addNode({b, node_data_b});

                     // Simplified corner handling
                    // graph.addNode({b, {b_t, {px_b, py_b, pz_b}, 
                        // {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, ghosted_b, interior_b}});
                } 
                else 
                {
                    graph.addNode({b, {b_t, {px_b, py_b, pz_b}, 
                        {{px_b, py_b, pz_b}, {px_b, py_b, pz_b}, {px_b, py_b, pz_b}, {px_b, py_b, pz_b}}, ghosted_b, interior_b}});
                    std::cout << "Adding node for cell b at (" << px_b << ", " << py_b << ", " << pz_b << ") of type " << b_t << "\n";
                    std::cout << "Graph now has " << graph.numNodes() << " nodes after adding cell b.\n";
                
                }
                
                std::cout << "Adding edge between " << a << " and " << b << "\n";
                std::cout << "a: (" << px_a << ", " << py_a << ", " << pz_a << "), type: " << a_t << ", ghosted: " << ghosted_a << "\n";
                std::cout << "b: (" << px_b << ", " << py_b << ", " << pz_b << "), type: " << b_t << ", ghosted: " << ghosted_b << "\n";
                graph.addEdge(a, b);

            }

            bool ghost_cells;
            double min_x, min_y, min_z, max_x, max_y, max_z;
            std::size_t num_interior_0D_cells;
            std::size_t num_exterior_0D_cells;
            std::size_t num_interior_1D_cells;
            std::size_t num_exterior_1D_cells;
            std::size_t num_interior_2D_cells;
            std::size_t num_exterior_2D_cells;
            std::size_t num_interior_3D_cells; // New
            std::size_t num_exterior_3D_cells; // New

            std::size_t total_cells;
            std::size_t nx; // number of cells in x direction
            std::size_t ny; // number of cells in y direction 
            std::size_t nz; // number of cells in z direction 
            
            std::size_t ppc; 

            double dx; 
            double dy; 
            double dz; // New
            
            CplexGraph3D_t graph; // Assuming this is defined
            CartesianGrid3D fine_grid; // Assuming this is defined
            CartesianGrid3D coarse_grid; // Assuming this is defined
    };

    } // end namespace DGGML

#endif
