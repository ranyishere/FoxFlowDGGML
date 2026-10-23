#ifndef CARTESIAN_HASH_FUNCTIONS_3D_HPP
#define CARTESIAN_HASH_FUNCTIONS_3D_HPP

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>

namespace DGGML3D
{

// --- 3D Volume Hash (Finds the coarse 3D cell ID) ---
template <typename NodeType, typename CplexType>
int64_t cartesian_complex_expanded_hash3D(NodeType& node, CplexType& geoplex3D)
{
    auto& fine_grid = geoplex3D.getFineGrid();
    auto& coarse_grid = geoplex3D.getCoarseGrid();
    auto& node_data = node.getData();
        
    double xp = node_data.position[0];
    double yp = node_data.position[1];
    double zp = node_data.position[2];
    int ic, jc, kc;

    coarse_grid.locatePoint(xp, yp, zp, ic, jc, kc); 
    
    // Bounds check
    if(ic < 0 || ic >= coarse_grid._nx || jc < 0 || jc >= coarse_grid._ny || kc < 0 || kc >= coarse_grid._nz) 
    {
        for(auto i = 0; i < 4; i++) node_data.tagND[i] = -1;
        std::cout << "out of bounds error in 3D phase\n"; 
        return -1;
    }

    // Map to fine grid cardinal index (Volume ID)
    // Note: Ensure your 3D complex has a method to map coarse coords to fine cardinal
    auto cardinal = fine_grid.cardinalLatticeIndex(ic, jc, kc); 
    node_data.tagND[0] = cardinal;

    return cardinal;
}

// --- 2D Face Hash (Checks neighbors of Volume for expanded Faces) ---
template <typename NodeType, typename CplexType>
int64_t cartesian_complex_expanded_hash2D(NodeType& node, CplexType& geoplex3D)
{
    auto& geoplex_graph = geoplex3D.getGraph();
    auto& node_data = node.getData();
    double xp = node_data.position[0], yp = node_data.position[1], zp = node_data.position[2];

    // Find Volume ID first
    auto cardinal = cartesian_complex_expanded_hash3D(node, geoplex3D);
    if(cardinal == -1) return -1;

    // Search neighbors of Volume for Face (type 1)
    for(auto jter = geoplex_graph.out_neighbors_begin(cardinal); jter != geoplex_graph.out_neighbors_end(cardinal); jter++)
    {
        auto jd = *jter;
        auto& jnode_data = geoplex_graph.findNode(jd)->second.getData();
        
        if(jnode_data.type == 1 && jnode_data.interior) 
        {
            auto& corners = jnode_data.corners;
            double xmin = corners[0][0], xmax = corners[0][0], ymin = corners[0][1], ymax = corners[0][1], zmin = corners[0][2], zmax = corners[0][2];
            for(int i = 1; i < 8; i++) {
                xmin = std::min(xmin, corners[i][0]); xmax = std::max(xmax, corners[i][0]);
                ymin = std::min(ymin, corners[i][1]); ymax = std::max(ymax, corners[i][1]);
                zmin = std::min(zmin, corners[i][2]); zmax = std::max(zmax, corners[i][2]);
            }

            if(xp >= xmin && xp <= xmax && yp >= ymin && yp <= ymax && zp >= zmin && zp <= zmax)
            {
                node_data.tagND[1] = jd;
                return jd;
            }
        }
    } 
    return cardinal;
}

// --- 1D Edge Hash (Checks neighbors of Face for expanded Edges) ---
template <typename NodeType, typename CplexType>
int64_t cartesian_complex_expanded_hash1D(NodeType& node, CplexType& geoplex3D)
{
    auto& geoplex_graph = geoplex3D.getGraph();
    auto& node_data = node.getData();
    double xp = node_data.position[0], yp = node_data.position[1], zp = node_data.position[2];

    // Find Face ID first
    auto face_id = cartesian_complex_expanded_hash2D(node, geoplex3D);
    if(face_id == -1) return -1;

    // Search neighbors of Face for Edge (type 2)
    for(auto jter = geoplex_graph.out_neighbors_begin(face_id); jter != geoplex_graph.out_neighbors_end(face_id); jter++)
    {
        auto jd = *jter;
        auto& jnode_data = geoplex_graph.findNode(jd)->second.getData();
        if(jnode_data.type == 2 && jnode_data.interior) 
        {
            auto& corners = jnode_data.corners;
            double xmin = corners[0][0], xmax = corners[0][0], ymin = corners[0][1], ymax = corners[0][1], zmin = corners[0][2], zmax = corners[0][2];
            for(int i = 1; i < 8; i++) {
                xmin = std::min(xmin, corners[i][0]); xmax = std::max(xmax, corners[i][0]);
                ymin = std::min(ymin, corners[i][1]); ymax = std::max(ymax, corners[i][1]);
                zmin = std::min(zmin, corners[i][2]); zmax = std::max(zmax, corners[i][2]);
            }
            if(xp >= xmin && xp <= xmax && yp >= ymin && yp <= ymax && zp >= zmin && zp <= zmax)
            {
                node_data.tagND[2] = jd;
                return jd;
            }
        }
    } 
    return face_id;
}

// --- 0D Vertex Hash (Checks neighbors of Edge for expanded Vertices) ---
template <typename NodeType, typename CplexType>
int64_t cartesian_complex_expanded_hash0D(NodeType& node, CplexType& geoplex3D)
{
    auto& geoplex_graph = geoplex3D.getGraph();
    auto& node_data = node.getData();
    double xp = node_data.position[0], yp = node_data.position[1], zp = node_data.position[2];

    // Find Edge ID first
    auto edge_id = cartesian_complex_expanded_hash1D(node, geoplex3D);
    if(edge_id == -1) return -1;

    // Search neighbors of Edge for Vertex (type 3)
    for(auto jter = geoplex_graph.out_neighbors_begin(edge_id); jter != geoplex_graph.out_neighbors_end(edge_id); jter++)
    {
        auto jd = *jter;
        auto& jnode_data = geoplex_graph.findNode(jd)->second.getData();
        if(jnode_data.type == 3 && jnode_data.interior) 
        {
            auto& corners = jnode_data.corners;
            double xmin = corners[0][0], xmax = corners[0][0], ymin = corners[0][1], ymax = corners[0][1], zmin = corners[0][2], zmax = corners[0][2];
            for(int i = 1; i < 8; i++) {
                xmin = std::min(xmin, corners[i][0]); xmax = std::max(xmax, corners[i][0]);
                ymin = std::min(ymin, corners[i][1]); ymax = std::max(ymax, corners[i][1]);
                zmin = std::min(zmin, corners[i][2]); zmax = std::max(zmax, corners[i][2]);
            }
            if(xp >= xmin && xp <= xmax && yp >= ymin && yp <= ymax && zp >= zmin && zp <= zmax)
            {
                node_data.tagND[3] = jd;
                return jd;
            }
        }
    }
    return edge_id;
}

// --- Updated 3D Sort Function ---
template <typename BucketType, typename ComplementType, typename GeoplexType, typename SysGraphType>
void expanded_cartesian_complex_sort_stl(BucketType& bucketsND, ComplementType& complementND,
		GeoplexType& geoplex3D, SysGraphType& system_graph)
{
    auto& geoplex_graph = geoplex3D.getGraph();
    
    // 1. Initialize buckets for all 4 dimensions (0=3D, 1=2D, 2=1D, 3=0D)
    for(auto iter = geoplex_graph.node_list_begin(); iter != geoplex_graph.node_list_end(); iter++)
    {
        auto id = iter->first;
        auto itype = iter->second.getData().type; // Mapping: 0=3D, 1=2D, 2=1D, 3=0D
        if(itype >= 0 && itype <= 3) { 
            bucketsND[itype].insert({id, {}});
        }
    }
    
    // 2. Map system graph nodes to topological buckets
    for(auto iter = system_graph.node_list_begin(); iter != system_graph.node_list_end(); iter++)
    {
        auto id = iter->first;
        auto& node = iter->second;
        
        for(auto i = 0; i < 4; i++) // Reset tagND for 4 dimensions
           node.getData().tagND[i] = -1;

        for(auto i = 0; i < 4; i++) 
        {
            std::size_t cardinal;
            if(i == 0) cardinal = cartesian_complex_expanded_hash3D(node, geoplex3D);
            if(i == 1) cardinal = cartesian_complex_expanded_hash2D(node, geoplex3D);
            if(i == 2) cardinal = cartesian_complex_expanded_hash1D(node, geoplex3D);
            if(i == 3) cardinal = cartesian_complex_expanded_hash0D(node, geoplex3D);

            auto search = bucketsND[i].find(cardinal);
            if(search != bucketsND[i].end())
                bucketsND[i].find(cardinal)->second.push_back(id);
            else
                complementND[i]++;
        }
    }
}

// --- Updated 3D Bucket Builder ---
template <typename BucketType, typename GeoplexType>
void build_bucketsND(BucketType& bucketsND, GeoplexType& geoplex3D)
{
    auto& geoplex_graph = geoplex3D.getGraph();
    
    for(auto iter = geoplex_graph.node_list_begin(); iter != geoplex_graph.node_list_end(); iter++)
    {
        auto id = iter->first;
        auto itype = iter->second.getData().type;
        auto interior = iter->second.getData().interior;

        if(itype >= 0 && itype <= 3 && interior) { 
            bucketsND[itype].insert({id, {}});
        }
    }

}

} // namespace DGGML

#endif
