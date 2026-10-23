#ifndef EXPANDED_CELL_COMPLEX_3D_HPP
#define EXPANDED_CELL_COMPLEX_3D_HPP

// Include the 3D base complex and 3D grid
#include "CartesianComplex3D.hpp" 
#include "CartesianGrid3D.hpp" 
// Assuming CartesianComplex3D uses CplexGraph3D_t
#include "CellComplexTypes.hpp" 

namespace DGGML
{
    // CplexGraph3D_t is defined in CellComplexTypes.hpp
    template <typename GraphType = CplexGraph3D_t>
    class ExpandedComplex3D : public CartesianComplex3D<GraphType>
    {
        public:
            
            using types = CartesianComplex3D<GraphType>;
            
            CartesianGrid3D reaction_grid; 
            
            // dim_label: 0=0D, 1=1D, 2=2D, 3=3D, 4=Ghost
            std::vector<int> dim_label; 
            std::vector<std::size_t> cell_label;

            ExpandedComplex3D() = default; 
            ExpandedComplex3D(std::size_t n, std::size_t m, std::size_t l, double d_x, double d_y, double d_z, bool g_c = false, double _epsilon = 0.0) 
            : CartesianComplex3D<GraphType>(n, m, l, d_x, d_y, d_z, g_c)
            {
                init(n, m, l, d_x, d_y, d_z, g_c, _epsilon);                 
            }     
            
           void init(std::size_t n, std::size_t m, std::size_t l, double d_x, double d_y, double d_z, bool g_c = false, double _epsilon = 0.0)
           {

                // 1. Initialize the base complex (CartesianComplex3D)
                CartesianComplex3D<GraphType>::init(n, m, l, d_x, d_y, d_z, g_c);
                
                if(_epsilon == 0.0)
                    // Set epsilon relative to the smallest cell width
                    epsilon = 0.003 * std::min({this->dx, this->dy, this->dz}); 
                else
                    epsilon = _epsilon;
                
                if(g_c)
                {
                    n = n+2; 
                    m = m+2; 
                    l = l+2;
                } 
                
                // 2. Initialize Reaction Grid
                // Calculate the number of reaction cells per coarse cell dimension
                std::size_t n_r_c = std::floor( ( this->dx ) / epsilon );
                std::size_t m_r_c = std::floor( ( this->dy ) / epsilon );
                std::size_t l_r_c = std::floor( ( this->dz ) / epsilon );
                
                auto n_r = n_r_c * n;
                auto m_r = m_r_c * m;
                auto l_r = l_r_c * l;

                // reaction_grid.init(min_x, min_y, min_z, max_x, max_y, max_z, nx, ny, nz)
                reaction_grid.init(0.0, 0.0, 0.0, n*d_x, m*d_y, l*d_z, n_r, m_r, l_r);

                // 3. Initialize label vectors
                dim_label.resize(reaction_grid.totalNumCells());
                cell_label.resize(reaction_grid.totalNumCells());
                
                // Default fill: 3D cells (d=3) will be the first and largest regions labeled.
                std::fill(dim_label.begin(), dim_label.end(), 0); 
                
                // 4. Build the expansion
                build();
           }
           
           GraphType& getGraph()
           {
               return this->graph;
           }
            
           void build() 
           {

                // Labeling order ensures precedence: 3D -> 2D -> 1D -> 0D -> Ghost
                // The later labels overwrite earlier ones, so 0D (Type 3) is done last.
                // 1. 3D Cells (Type 0) - Interior Volume (Lowest Precedence)
                for(auto iter = this->graph.node_list_begin(); iter != this->graph.node_list_end(); iter++) 
                {
                    auto key = iter->first;
                    auto& data = iter->second.getData();
                    
                    if( data.type == 0 && data.interior == true) // Interior 3D cell (Volume)
                    {
                        // Use the coarse cell boundaries stored in the node data (assuming 8 corners/bounding box)
                        double llb_x, llb_y, llb_z, urf_x, urf_y, urf_z;
                        
                        // Assuming CartesianComplex3D stores the bounding box at corners[0] and corners[7]
                        // We must rely on the coordinates derived from the fine grid centers (ppc=2)
                        int i_c, j_c, k_c;
                        this->fine_grid.pointToCardinalLattice(data.position[0], data.position[1], data.position[2], i_c, j_c, k_c);

                        // Lower-left-back corner: (i_c-1, j_c-1, k_c-1)
                        this->fine_grid.cardinalLatticeToPoint(llb_x, llb_y, llb_z, this->fine_grid.cardinalLatticeIndex(i_c-1, j_c-1, k_c-1));
                        
                        // Upper-right-front corner: (i_c+1, j_c+1, k_c+1)
                        this->fine_grid.cardinalLatticeToPoint(urf_x, urf_y, urf_z, this->fine_grid.cardinalLatticeIndex(i_c+1, j_c+1, k_c+1));

                        // Label the entire coarse cell volume (d=3 for 3D)
                        label(llb_x, llb_y, llb_z, urf_x, urf_y, urf_z, 0, key);
                    }

                }
                
                // 2. 2D Cells (Type 1) - Faces
                // Expansion is 1D orthogonal to the face by +/- gamma_epsilon
                auto gamma_epsilon_1d = 2.0 * epsilon; 

                for(auto iter = this->graph.node_list_begin(); iter != this->graph.node_list_end(); iter++) 
                {
                    auto key = iter->first;
                    auto& data = iter->second.getData();
                    
                    if(data.type == 1 && data.interior == true) // Interior 2D cell (Face)
                    {
                        double x, y, z;
                        int i_c, j_c, k_c;
                        x = data.position[0]; y = data.position[1]; z = data.position[2];
                        this->fine_grid.pointToCardinalLattice(x, y, z, i_c, j_c, k_c);
                        
                        double min_x, min_y, min_z, max_x, max_y, max_z;
                        
                        // Determine the face orientation (one index is even, two are odd)
                        
                        if (i_c % 2 == 0) // Face is parallel to YZ plane (Expand in X)
                        {
                            min_x = x - gamma_epsilon_1d; max_x = x + gamma_epsilon_1d;
                            
                            // Y and Z bounds from adjacent coarse cell boundaries (ppc=2 means distance is 1 fine grid unit)
                            this->fine_grid.cardinalLatticeToPoint(min_y, min_z, this->fine_grid.cardinalLatticeIndex(i_c, j_c - 1, k_c - 1));
                            this->fine_grid.cardinalLatticeToPoint(max_y, max_z, this->fine_grid.cardinalLatticeIndex(i_c, j_c + 1, k_c + 1));
                        } 
                        else if (j_c % 2 == 0) // Face is parallel to XZ plane (Expand in Y)
                        {
                            min_y = y - gamma_epsilon_1d; max_y = y + gamma_epsilon_1d;
                            
                            this->fine_grid.cardinalLatticeToPoint(min_x, min_z, this->fine_grid.cardinalLatticeIndex(i_c - 1, j_c, k_c - 1));
                            this->fine_grid.cardinalLatticeToPoint(max_x, max_z, this->fine_grid.cardinalLatticeIndex(i_c + 1, j_c, k_c + 1));
                        } 
                        else if (k_c % 2 == 0) // Face is parallel to XY plane (Expand in Z)
                        {
                            min_z = z - gamma_epsilon_1d; max_z = z + gamma_epsilon_1d;
                            
                            this->fine_grid.cardinalLatticeToPoint(min_x, min_y, this->fine_grid.cardinalLatticeIndex(i_c - 1, j_c - 1, k_c));
                            this->fine_grid.cardinalLatticeToPoint(max_x, max_y, this->fine_grid.cardinalLatticeIndex(i_c + 1, j_c + 1, k_c));
                        } else {
                            continue; // Should not happen for face centers
                        }
                        
                        // Label the expanded face region (d=1 for 2D)
                        label(min_x, min_y, min_z, max_x, max_y, max_z, 1, key);
                    }

                }
                
                // 3. 1D Cells (Type 2) - Edges
                // Expansion is 2D orthogonal to the edge by +/- gamma_epsilon
                auto gamma_epsilon_2d = 2.0 * epsilon; 
                for(auto iter = this->graph.node_list_begin(); iter != this->graph.node_list_end(); iter++) 
                {
                    auto key = iter->first;
                    auto& data = iter->second.getData();
                    
                    if(data.type == 2 && data.interior == true) // Interior 1D cell (Edge)
                    {
                        double x, y, z;
                        int i_c, j_c, k_c;
                        x = data.position[0]; y = data.position[1]; z = data.position[2];
                        this->fine_grid.pointToCardinalLattice(x, y, z, i_c, j_c, k_c);
                        
                        double min_x, min_y, min_z, max_x, max_y, max_z;
                        
                        // Determine the edge orientation (one index is odd, two are even)
                        
                        if (i_c % 2 != 0) // Edge is parallel to X axis (Expand in Y and Z)
                        {
                            min_y = y - gamma_epsilon_2d; max_y = y + gamma_epsilon_2d;
                            min_z = z - gamma_epsilon_2d; max_z = z + gamma_epsilon_2d;
                            
                            // X bounds fixed by adjacent coarse cell boundaries
                            this->fine_grid.cardinalLatticeToPoint(min_x, y, z, this->fine_grid.cardinalLatticeIndex(i_c - 1, j_c, k_c));
                            this->fine_grid.cardinalLatticeToPoint(max_x, y, z, this->fine_grid.cardinalLatticeIndex(i_c + 1, j_c, k_c));
                        } 
                        else if (j_c % 2 != 0) // Edge is parallel to Y axis (Expand in X and Z)
                        {
                            min_x = x - gamma_epsilon_2d; max_x = x + gamma_epsilon_2d;
                            min_z = z - gamma_epsilon_2d; max_z = z + gamma_epsilon_2d;

                            this->fine_grid.cardinalLatticeToPoint(x, min_y, z, this->fine_grid.cardinalLatticeIndex(i_c, j_c - 1, k_c));
                            this->fine_grid.cardinalLatticeToPoint(x, max_y, z, this->fine_grid.cardinalLatticeIndex(i_c, j_c + 1, k_c));
                        } 
                        else if (k_c % 2 != 0) // Edge is parallel to Z axis (Expand in X and Y)
                        {
                            min_x = x - gamma_epsilon_2d; max_x = x + gamma_epsilon_2d;
                            min_y = y - gamma_epsilon_2d; max_y = y + gamma_epsilon_2d;
                            
                            this->fine_grid.cardinalLatticeToPoint(x, y, min_z, this->fine_grid.cardinalLatticeIndex(i_c, j_c, k_c - 1));
                            this->fine_grid.cardinalLatticeToPoint(x, y, max_z, this->fine_grid.cardinalLatticeIndex(i_c, j_c, k_c + 1));
                        } else {
                            continue; // Should not happen for edge centers
                        }
                        
                        // Label the expanded edge region (d=1 for 1D)
                        label(min_x, min_y, min_z, max_x, max_y, max_z, 2, key);
                    }
                }

                // 4. 0D Cells (Type 3) and Ghost 3D Cells (Type 0, Exterior) - Highest Precedence
                for(auto iter = this->graph.node_list_begin(); iter != this->graph.node_list_end(); iter++) 
                {
                    auto key = iter->first;
                    auto& data = iter->second.getData();
                    
                    if( data.type == 0 && data.interior == false) // Exterior 3D Ghost cell
                    {
                        // Same bounding box logic as interior 3D cell
                        double llb_x, llb_y, llb_z, urf_x, urf_y, urf_z;
                        int i_c, j_c, k_c;
                        
                        this->fine_grid.pointToCardinalLattice(data.position[0], data.position[1], data.position[2], i_c, j_c, k_c);
                        this->fine_grid.cardinalLatticeToPoint(llb_x, llb_y, llb_z, this->fine_grid.cardinalLatticeIndex(i_c-1, j_c-1, k_c-1));
                        this->fine_grid.cardinalLatticeToPoint(urf_x, urf_y, urf_z, this->fine_grid.cardinalLatticeIndex(i_c+1, j_c+1, k_c+1));
                        
                        // Label with d=4 to distinguish ghost cells from interior cells
                        label(llb_x, llb_y, llb_z, urf_x, urf_y, urf_z, 4, key); 
                    } 
                    
                    if( data.type == 3 && data.interior == true) // Interior 0D cell (Vertex)
                    {
                        // Expands in 3D around the point
                        double cx = data.position[0];
                        double cy = data.position[1];
                        double cz = data.position[2];
                        
                        // Use a larger expansion factor
                        auto gamma_epsilon_3d = 3.5*epsilon; 
                        
                        double min_x, min_y, min_z, max_x, max_y, max_z;
                        min_x = cx - gamma_epsilon_3d; max_x = cx + gamma_epsilon_3d;
                        min_y = cy - gamma_epsilon_3d; max_y = cy + gamma_epsilon_3d;
                        min_z = cz - gamma_epsilon_3d; max_z = cz + gamma_epsilon_3d;
                        
                        // Label the expanded vertex region (d=3 for 3D)
                        label(min_x, min_y, min_z, max_x, max_y, max_z, 3, key);
                    } 
                }
           }
            

           // 3D Labeling function
           void label(double ll_x, double ll_y, double ll_z,
           double ur_x, double ur_y, double ur_z, int d, std::size_t key) 
           {
                int min_i, min_j, min_k, max_i, max_j, max_k;

                // Locate the bounding box coordinates in the reaction grid
                reaction_grid.locatePoint(ll_x, ll_y, ll_z, min_i, min_j, min_k);
                reaction_grid.locatePoint(ur_x, ur_y, ur_z, max_i, max_j, max_k);
                
                // Clamp indices to be within the grid bounds
                min_i = std::max(0, min_i);
                min_j = std::max(0, min_j);
                min_k = std::max(0, min_k);
                max_i = std::min((int)reaction_grid._px - 1, max_i);
                max_j = std::min((int)reaction_grid._py - 1, max_j);
                max_k = std::min((int)reaction_grid._pz - 1, max_k);

                // Loop over all reaction cells within the bounding box and assign label
                for(auto i = min_i; i <= max_i; i++)
                {
                    for(auto j = min_j; j <= max_j; j++)
                    {
                        for(auto k = min_k; k <= max_k; k++)
                        {
                            auto cardinal = reaction_grid.cardinalCellIndex(i, j, k);

                            if (d > dim_label[cardinal] || cell_label[cardinal] == 0) {
                                dim_label[cardinal] = d;
                                cell_label[cardinal] = key;

                            // std::cout << "Labeled reaction cell at (" << i << ", " << j << ", " << k << ") with dim " << d << " and key " << key << "\n";
                            }

                            // dim_label[cardinal] = d;
                            // cell_label[cardinal] = key;
                        }
                    }
                }

           }
            
            // Output stream operator
            friend std::ostream& operator<<(std::ostream& os, ExpandedComplex3D& geoplex)
            {
                os << "\n---Expanded Complex 3D---\n";
                os << "\nCoarse Grid: \n" << geoplex.coarse_grid;
                os << "\nFine Grid: \n" << geoplex.fine_grid;
                os << geoplex.graph;
                
                return os;
            }
             
        private:
           double epsilon;

        };

} // ending namespace DGGML

#endif
