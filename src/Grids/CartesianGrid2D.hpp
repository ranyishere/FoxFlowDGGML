#ifndef __DGGML_CARTESIANGRID2D_HPP
#define __DGGML_CARTESIANGRID2D_HPP

#include "CartesianGrid3D.hpp"

namespace DGGML
{

/**
 * @brief 2D Grid specialized from the 3D implementation.
 * It treats the 2D plane as a 3D grid with 1 cell in the Z direction.
 */
class CartesianGrid2D : public CartesianGrid3D {

    public:
        CartesianGrid2D() : CartesianGrid3D() {}

        // Specialized 2D Init: Forces Z to be a flat slice
        void init(const double min_x, const double min_y,
                  const double max_x, const double max_y,
                  const std::size_t nx, const std::size_t ny) 
        {
            // Call 3D init with Z fixed at 0.0 to 1.0 (or 0.0 to 0.0) 
            // and 1 cell in Z direction.
            CartesianGrid3D::init(min_x, min_y, 0.0, 
                                  max_x, max_y, 0.0, 
                                  nx, ny, 1);
        }

        // Overload to maintain original 2D API for locating points
        void locatePoint(const double xp, const double yp, int& ic, int& jc) const {
            int kc;
            CartesianGrid3D::locatePoint(xp, yp, 0.0, ic, jc, kc);
        }

        // Cell indexing mapping to 3D logic (where k=0)
        void ijCellIndex(const int cardinal, int& ic, int& jc) const {
            int kc;
            CartesianGrid3D::ijkCellIndex(cardinal, ic, jc, kc);
        }

        int cardinalCellIndex(const int i, const int j) const {
            return CartesianGrid3D::cardinalCellIndex(i, j, 0);
        }

        // Coordinate conversion for cell centers
        void cardinalCellToPoint(double& xp, double& yp, const int cardinal) const {
            double zp;
            CartesianGrid3D::cardinalCellToPoint(xp, yp, zp, cardinal);
        }

        int cardinalLatticeIndex(const int i, const int j) const {
            return CartesianGrid3D::cardinalLatticeIndex(i, j, 0);
        } 

        int cardinallCellIndex(const int i, const int j) const {
            return CartesianGrid3D::cardinalCellIndex(i, j, 0);
        }

        // Print override to show 2D specific info
        friend std::ostream& operator<<(std::ostream& os, CartesianGrid2D& grid) {
            os << "Grid2D (via Grid3D): {\n"
               << "\t{ bounds: [" << grid._min_x << "," << grid._max_x << "] x [" 
               << grid._min_y << "," << grid._max_y << "] }\n"
               << "\t{ resolution: " << grid._nx << "x" << grid._ny << " }\n"
               << "\t{ cell_size: " << grid._dx << "x" << grid._dy << " }\n"
               << "}\n";
            return os;
        }
};

} // end namespace DGGML

#endif
