#ifndef CELL_LIST_HPP
#define CELL_LIST_HPP

#include <vector>

#include "CartesianGrid2D.hpp"
#include "CartesianGrid3D.hpp"
#include "CartesianGrid.hpp"
#include "ComponentMatchMap.hpp"
#include "YAGL_Graph.hpp"
#include "Listeners.hpp"
#include <omp.h>


namespace DGGML
{

//type for holding the CellList Data in CSR format
//could be customized for various key types
using ViewType = std::vector<std::size_t>;

//This is the CSR layout version
struct CellListDataCSR
{
    //number of objects per cell
    ViewType counts; 

    //offsets for cells 
    ViewType offsets;

    //keys associated with a cell, indexed by offsets
    ViewType permutation;

};

//CellList Data in a noncontiguous binned format
struct CellListDataVec
{
};

template <typename GraphType>
struct CellList
{

    //TODO: maybe we need a multidimensional iterator, to iterate over the outer and inner
    //CellListData data;
    using ComponentMapType = ComponentMatchMap<typename GraphType::key_type>;
    using ComponentMapIter = typename ComponentMapType::iterator;
    using CellListObjectType = typename ComponentMapType::key_type;
    using CellListDataType = std::vector<std::vector<CellListObjectType>>;
    //using GraphType = YAGL::Graph<Plant::mt_key_type, Plant::MT_NodeData>;

    using outer_iterator = typename CellListDataType::iterator;
    using inner_iterator = typename std::vector<CellListObjectType>::iterator;

    std::vector<Listener*> listeners;

    // One lock per reaction-grid bucket. Threads operating on spatially
    // disjoint buckets can insert/erase concurrently without serializing.
    std::vector<omp_lock_t> bucket_locks;

    struct IteratorPair
    {
        typename CellListDataType::iterator outer_iterator;
        typename std::vector<CellListObjectType>::iterator inner_iterator;
    };

    using iterator_pair = IteratorPair;
    CellListDataType data;

    // CartesianGrid2D grid;
    CartesianGrid3D grid;

    GraphType* graph;
    ComponentMapType* component_matches;
    int cell_range;

    bool is_3D = false;

    void toggle_3D(bool val) {
        is_3D = val;
    }

    ~CellList() {
        // Cleaning up listeners if cell list is gone.
        listeners.clear();
        for (auto& lk : bucket_locks)
            omp_destroy_lock(&lk);
    }

    CellList(CartesianGrid3D & _grid, GraphType* _graph, ComponentMapType* _component_matches, bool is_3D = false, double interaction_radius = 0.0, double geocell_size = 0.0)
        : grid(_grid), graph(_graph), component_matches(_component_matches), data(_grid.totalNumCells()), is_3D(is_3D)
    {
        // Initialise one OMP lock per reaction-grid bucket.
        bucket_locks.resize(_grid.totalNumCells());
        for (auto& lk : bucket_locks)
            omp_init_lock(&lk);
        // Compute cell_range: number of reaction-grid cells to search in each direction.
        // With cell_range=r, the stencil covers (2r+1)^D cells.
        //
        // Strategy: cell_range must cover the interaction_radius, but should never
        // exceed one geocell's worth of reaction cells. The expanded cell complex
        // already ensures interacting components lie within adjacent geocells, so
        // searching beyond one geocell is redundant.

        if (interaction_radius > 0.0 && _grid._dx > 0.0) {
            int needed = (int)std::ceil(interaction_radius / _grid._dx);
            if (geocell_size > 0.0) {
                // Cap at one geocell's width in reaction-grid cells
                int one_geocell = (int)std::floor(geocell_size / _grid._dx);
                cell_range = std::min(needed, std::max(1, one_geocell));
            } else {
                cell_range = needed;
            }
            cell_range = std::max(1, cell_range);
        } else {
            cell_range = 1;
        }

	// cell_range = 3;
        std::cout << "CellList: cell_range=" << cell_range 
                  << " (grid_dx=" << _grid._dx 
                  << ", interaction_radius=" << interaction_radius
                  << ", geocell_size=" << geocell_size << ")\n";

        //build the cell list from the system graph 
//        std::cout << "Building cell list\n";
        for(auto it = component_matches->begin(); it != component_matches->end(); it++) insert_match(it);
//        std::cout << "Cell list has been built\n";
//        std::cout << "The cell list has " << totalNumCells() << " reaction cells\n";
//        std::cout << "Components mapped: " << getTotalSize() << "\n";
    }
    
    std::size_t getCellSize(const int cell)
    {
        return data[cell].size();
    }
    
    std::size_t getTotalSize()
    {
        auto sum = 0;
        for(const auto& cell : data) sum += cell.size();
        return sum;
    }

    std::size_t locate_cell(ComponentMapIter& comp_iter)
    {

        if (this->is_3D) {
            return locate_cell_3d(comp_iter);
        }

        auto& match = comp_iter->second;
        auto& anchor_data = graph->findNode(match.anchor)->second.getData();
        double xp = anchor_data.position[0];
        double yp = anchor_data.position[1];
        auto key = comp_iter->first;
        int ic, jc;
        grid.locatePoint(xp, yp, ic, jc);
        return grid.cardinalCellIndex(ic, jc);
    }

    std::size_t locate_cell_3d(ComponentMapIter& comp_iter) {
        auto& match = comp_iter->second;
        auto& anchor_data = graph->findNode(match.anchor)->second.getData();
        double xp = anchor_data.position[0];
        double yp = anchor_data.position[1];
        double zp = anchor_data.position[2];

        auto key = comp_iter->first;
        int ic, jc, kc;
        grid.locatePoint(xp, yp, zp, ic, jc, kc);
        return grid.cardinalCellIndex(ic, jc, kc);
    }

    void insert_match(ComponentMapIter& comp_iter)
    {
        // Handles 3D case
        if (is_3D == 1) {
            auto cardinal = locate_cell_3d(comp_iter);
            omp_set_lock(&bucket_locks[cardinal]);
            data[cardinal].push_back(comp_iter->first);
            omp_unset_lock(&bucket_locks[cardinal]);
            return;
        }

        auto cardinal = locate_cell(comp_iter);
        omp_set_lock(&bucket_locks[cardinal]);
        data[cardinal].push_back(comp_iter->first);
        omp_unset_lock(&bucket_locks[cardinal]);
        this->notify();
    }


    /**
     * 2D Version Need to use only the 3D version.
     * */
    //The find function must search the current cell and potentially it's neigbhors to see where a component is
    /*
    iterator_pair find(ComponentMapIter& comp_iter)
    {
        //first see if the component is in the cell it would be currently mapped to
        auto c = locate_cell(comp_iter);
        bool found = false;
        for(auto i = 0; i < data[c].size(); i++)
        {
            if(comp_iter->first == data[c][i]) {
                found = true;
//                std::cout << "Found component " << comp_iter->first << " in cell " << c << " at index " << i << "\n";
                auto outIter = data.begin()+c;
                auto inIter = outIter->begin()+i;
                return iterator_pair{outIter, inIter};
            }
        }

        if(!found) //expand the search, since currently mapped cell, may not be the original (components move)
        {
            int imin, imax, jmin, jmax;
            getCells(c, imin, imax, jmin, jmax);
            for(auto i = imin; i < imax; i++)
            {
                for(auto j = jmin; j < jmax; j++)
                {
                    auto d = grid.cardinalCellIndex(i, j);
//                    std::cout << "Searching in cell " << d << "\n";
                    //if(d == c) continue;
                    for(auto k = 0; k < data[d].size(); k++)
                    {
                        if(comp_iter->first == data[d][k]) {
                            found = true;
//                            std::cout << "Found component " << comp_iter->first << " in cell " << d << " at index " << k << "\n";
                            auto outIter = data.begin()+d;
                            auto inIter = outIter->begin()+k;
                            return iterator_pair{outIter, inIter};
                        }
                    }
                }
            }
        }
        auto outIter = data.end();
        auto inIter = outIter->end();
        return iterator_pair {outIter, inIter};
    }
    */


    /**
     * 3-D Version
     * */
    iterator_pair find(ComponentMapIter& comp_iter)
    {
        // First see if the component is in the cell it would be currently mapped to

        std::size_t c;
        if (is_3D)
        c = locate_cell_3d(comp_iter);
        else 
        c = locate_cell(comp_iter);

        bool found = false;

        // Search within the primary cell
        for(auto i = 0; i < data[c].size(); i++)
        {
            if(comp_iter->first == data[c][i]) {
                found = true;
                auto outIter = data.begin() + c;
                auto inIter = outIter->begin() + i;
                return iterator_pair{outIter, inIter};
            }
        }

        // Expand the search to the surrounding stencil if not found in the primary cell
        if(!found)
        {
            int imin, imax, jmin, jmax, kmin, kmax;
            // Updated call to get 3D bounds
            getCells(c, imin, imax, jmin, jmax, kmin, kmax);

            for(auto i = imin; i < imax; i++)
            {
                for(auto j = jmin; j < jmax; j++)
                {
                    // Added the third loop for the Z-dimension (k-index)
                    for(auto k = kmin; k < kmax; k++)
                    {
                        // Call the 3-argument version of cardinalCellIndex
                        auto d = grid.cardinalCellIndex(i, j, k);

                        for(auto m = 0; m < data[d].size(); m++)
                        {
                            if(comp_iter->first == data[d][m]) {
                                found = true;
                                auto outIter = data.begin() + d;
                                auto inIter = outIter->begin() + m;
                                return iterator_pair{outIter, inIter};
                            }
                        }
                    }
                }
            }
        }

        auto outIter = data.end();
        return iterator_pair {outIter, outIter != data.begin() ? outIter[-1].end() : data.begin()->end()};
    }

    void erase(ComponentMapIter& comp_iter)
    {
        // Determine the bucket first (read-only locate is safe without a lock),
        // then hold the bucket lock for the find+erase pair to close the TOCTOU gap.
        std::size_t cardinal;
        if (is_3D)
            cardinal = locate_cell_3d(comp_iter);
        else
            cardinal = locate_cell(comp_iter);

        omp_set_lock(&bucket_locks[cardinal]);
        auto r = find(comp_iter);
        if(r.outer_iterator != data.end())
            r.outer_iterator->erase(r.inner_iterator);
        omp_unset_lock(&bucket_locks[cardinal]);
        this->notify();
    }

    void rebuild()
    {
        //        std::cout << "Rebuilding cell list\n";
        for(auto& cell : data)
            cell.clear();
        //        std::cout << "Cleared cell list mapped components: " << getTotalSize() << "\n";

        // std::cout << "size: " << component_matches->size() << std::endl;
        for(auto it = component_matches->begin(); it != component_matches->end(); it++) {
            // std::cout << "Reinserting component " << it->first << " into cell list\n";
            // print out match
            // std::cout << "Match: { anchor: " << it->second.anchor << ", type: " << it->second.type << ", match: [";
            // std::cout << it->second.match[0];
            // for (size_t i = 1; i < it->second.match.size(); i++) {
                // std::cout << ", " << it->second.match[i];
            // }
            // std::cout << "] }\n";

            insert_match(it);
        }
       // std::cout << "Cell list has been built\n";
       // std::cout << "The cell list has " << totalNumCells() << " reaction cells\n";
       // std::cout << "Components mapped after reset: " << getTotalSize() << "\n";
        this->notify();
    }

    std::size_t totalNumCells() { return grid.totalNumCells(); }
    
    // 2D version
    std::size_t cardinalCellIndex(int i, int j) { return grid.cardinalCellIndex(i, j); }

    // 3D version
    std::size_t cardinalCellIndex(int i, int j, int k) { return grid.cardinalCellIndex(i, j, k); }

    // give a cell, get it's stencil
    void getCells(const int cell, int& imin, int& imax, int& jmin, int& jmax) 
    {
        // Temporary variables for 3D bounds but not needed here
        int kmin, kmax;

        this->getCells(cell, imin, imax, jmin, jmax, kmin, kmax);
    }

    // Given a cell, get its 3D stencil bounds
    void getCells(const int cell,
                  int& imin, int& imax,
                  int& jmin, int& jmax,
                  int& kmin, int& kmax)
    {
        int i, j, k;

        // Use the 3D indexing method from CartesianGrid3D
        grid.ijkCellIndex(cell, i, j, k);

        // X-direction bounds
        imin = (i - cell_range > 0) ? i - cell_range : 0;
        imax = (i + cell_range + 1 < grid._nx) ? i + cell_range + 1 : grid._nx;

        // Y-direction bounds
        jmin = (j - cell_range > 0) ? j - cell_range : 0;
        jmax = (j + cell_range + 1 < grid._ny) ? j + cell_range + 1 : grid._ny;

        // Z-direction bounds
        kmin = (k - cell_range > 0) ? k - cell_range : 0;
        kmax = (k + cell_range + 1 < grid._nz) ? k + cell_range + 1 : grid._nz;
    }

    void addListener(Listener* listener)
    {
        listeners.push_back(listener);
    }

    /**
     * Notify listeners
     */
    void notify()
    {
        for(auto listener : listeners)
            listener->onUpdate();
    }

    /**
     * Save the cell list to a file to be loaded
     */
    void save(const std::string& filename)
    {
    }

};

template <typename GraphType>
class CellListListener : public Listener
{
public:
    CellList<GraphType>* cell_list;

    CellListListener(CellList<GraphType>* _cell_list) : cell_list(_cell_list) {

        filename = "cell_listener.txt"; // Name of the file to open/create
        // Open the file with 'out' mode (for writing) and 'trunc' mode (to truncate/clear)
        // If the file does not exist, it will be created.
        // If the file exists, its content will be cleared.
        std::ofstream outfile(filename, std::ios::out | std::ios::trunc);

        // Check if the file was successfully opened
        if (outfile.is_open()) {
            outfile << "This is the first line." << std::endl;
            outfile << "This is the second line." << std::endl;
            outfile.close(); // Close the file stream
            std::cout << "Successfully wrote to " << filename << std::endl;
        } else {
            std::cerr << "Error: Unable to open or create " << filename << std::endl;
        }

    }

    std::string filename;

    void onUpdate() override
    {
        this->getInfo();
    }

    void getInfo() override
    {
        // Print out CellList Information
        std::ofstream outfile(this->filename, std::ios::app);
        // Check if the file was successfully opened
        if (outfile.is_open()) {
            outfile << "CellListListener: CellList has " << cell_list->totalNumCells() << " cells\n";
            outfile << "CellListListener: CellList has " << cell_list->getTotalSize() << " mapped components\n";
            outfile.close(); // Close the file stream
        } else {
            std::cerr << "Error: Unable to open or create " << filename << std::endl;
        }

    }

};

}
#endif 
