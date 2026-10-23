#ifndef __DGGML_CARTESIANGRID3D_HPP
#define __DGGML_CARTESIANGRID3D_HPP

#include <cmath>
#include <iostream>
#include "CartesianGrid.hpp"
#include <algorithm> // for std::min, std::max

namespace DGGML
{

class CartesianGrid3D : public CartesianGrid {
    public:
        double _min_x;
        double _min_y;
        double _min_z;
        double _max_x;
        double _max_y;
        double _max_z;
        double _dx;
        double _dy;
        double _dz;
        double _rdx; // reciprocal of dx
        double _rdy; // reciprocal of dy
        double _rdz; // reciprocal of dz
        int _nx; // number of cells in the x direction
        int _ny; // number of cells in the y direction
        int _nz; // number of cells in the z direction
        int _px; // number of grid points in the x direction 
        int _py; // number of grid points in the y direction
        int _pz; // number of grid points in the z direction

        CartesianGrid3D () {}
       
        void init(const double min_x, const double min_y, const double min_z,
                  const double max_x, const double max_y, const double max_z,
                  const std::size_t nx, const std::size_t ny, const std::size_t nz) 
        {
            _min_x = min_x;
            _min_y = min_y;
            _min_z = min_z;
            _max_x = max_x;
            _max_y = max_y;
            _max_z = max_z;
            
            _nx = nx; 
            _ny = ny;
            _nz = nz;
            _px = nx + 1;
            _py = ny + 1;
            _pz = nz + 1;

            _dx = (max_x - min_x) / _nx;
            _dy = (max_y - min_y) / _ny;
            _dz = (max_z - min_z) / _nz;

            _rdx = 1.0 / _dx;
            _rdy = 1.0 / _dy;
            _rdz = 1.0 / _dz;
        }
        
        // total number of cells formed from the lattice
        int totalNumCells() const { return _nx * _ny * _nz; }
       
        // total number of lattice grid points
        int totalNumPoints() const { return _px * _py * _pz; }

        int cellsBetween(const double max, const double min, const double rdelta) const 
        {
            return std::floor((max-min)*rdelta);
        }

        void locatePoint(double xp, double yp, int& ic, int& jc) const override {
            // Just assuming its a 2D call, so set z=0
            int kc;
            locatePoint(xp, yp, 0.0, ic, jc, kc); // Just call your 3D version
        }
        
        // locate which cell the point belongs to
        void locatePoint( const double xp, const double yp, const double zp, int& ic, int& jc, int& kc ) const
        {
            // Calculate indices
            ic = cellsBetween( xp, _min_x, _rdx );
            jc = cellsBetween( yp, _min_y, _rdy );
            kc = cellsBetween( zp, _min_z, _rdz );
            
            // Clamp indices to prevent out-of-bounds errors
            ic = std::max(0, std::min(ic, _nx - 1));
            jc = std::max(0, std::min(jc, _ny - 1));
            kc = std::max(0, std::min(kc, _nz - 1));
        }

        // Overload for 2D cell index support
        int cardinalCellIndex(const int i, const int j) const
        {
            // Simply call the 3D version with k=0
            return cardinalCellIndex(i, j, 0);
        }

        // Overload for 2D lattice index support
        int cardinalLatticeIndex(const int i, const int j) const
        {
            // Simply call the 3D version with k=0
            return cardinalLatticeIndex(i, j, 0);
        }
        
        // returns the ijk index of a cell from a cardinal index 
        void ijkCellIndex(const int cardinal, int& ic, int& jc, int& kc) const 
        {
           kc = cardinal / (_nx * _ny);
           int remainder = cardinal % (_nx * _ny);
           jc = remainder / _nx;
           ic = remainder % _nx;
        }
        
        // returns the ijk index of lattice point from a cardinal index
        void ijkLatticeIndex(const int cardinal, int& ic, int& jc, int& kc) const 
        {
            kc = cardinal / (_px * _py);
            int remainder = cardinal % (_px * _py);
            jc = remainder / _px;
            ic = remainder % _px;
        }

        // returns the cardinal cell index from ijk (using nx, ny)
        int cardinalCellIndex(const int i, const int j, const int k) const
        {
            // Formula: i + nx*j + (nx*ny)*k
            return i + _nx * (j + _ny * k);
        }
        
        // returns the cardinal lattice point index from ijk (using px, py)
        int cardinalLatticeIndex(const int i, const int j, const int k) const 
        {
            // Formula: i + px*j + (px*py)*k
            return i + _px * (j + _py * k);
        }

        // converts the cardinal index of a cell to its central point (x, y, z)
        void cardinalCellToPoint(double& xp, double& yp, double& zp, const int cardinal) const 
        {
            int i, j, k;
            ijkCellIndex(cardinal, i, j, k);

            xp = _min_x + i*_dx + _dx/2.0;
            yp = _min_y + j*_dy + _dy/2.0;
            zp = _min_z + k*_dz + _dz/2.0;
        }
        
        // converts the cardinal index of a lattice point to its exact point (x, y, z)
        void cardinalLatticeToPoint(double& xp, double& yp, double& zp, const int cardinal) const 
        {
            int i, j, k;
            ijkLatticeIndex(cardinal, i, j, k);
            xp = _min_x + i*_dx;
            yp = _min_y + j*_dy;
            zp = _min_z + k*_dz;
        }
        
        // converts the ijk indices of a lattice point to its exact point (x, y, z)
        void cardinalLatticeToPoint(double& xp, double& yp, double& zp, int i, int j, int k) const 
        {
            xp = _min_x + i*_dx;
            yp = _min_y + j*_dy;
            zp = _min_z + k*_dz;
        }

        // Overload: Converts the cardinal index of a lattice point to two exact points (e.g., y and z)
        // This is used for finding bounds in 3D when one axis is fixed (e.g., finding the Y-Z plane bounds)
        void cardinalLatticeToPoint(double& p1, double& p2, const int cardinal) const
        {
            int i, j, k;
            ijkLatticeIndex(cardinal, i, j, k);

            // Determine which two points correspond to p1 and p2 based on the 3D cell type logic.
            // Since the calling context (ExpandedComplex3D) is looking up Y and Z (in the X-face section),
            // we assume p1=Y and p2=Z, but the logic should be robust.
            // For safety and compatibility with the three-argument version, we explicitly calculate all three and return the last two.
            double x_coord, y_coord, z_coord;

            // Calculate the 3D coordinates
            x_coord = _min_x + i*_dx;
            y_coord = _min_y + j*_dy;
            z_coord = _min_z + k*_dz;

            // Assign to the two references provided. The calling code expects these to be Y and Z, respectively.
            p1 = y_coord;
            p2 = z_coord;
        }

        // Converts a physical point (x, y, z) to the nearest lattice point indices (i, j, k)
        // Note: This logic assumes the point is *on* the fine grid or used for a cell center lookup (as in ExpandedComplex3D)
        void pointToCardinalLattice(const double xp, const double yp, const double zp, int& ic, int& jc, int& kc) const
        {
            // For a lattice point, the index is calculated by dividing the distance from the minimum
            // boundary by the cell width (dx, dy, dz) and rounding to the nearest integer.

            // X index
            ic = std::round((xp - _min_x) * _rdx);

            // Y index
            jc = std::round((yp - _min_y) * _rdy);

            // Z index
            kc = std::round((zp - _min_z) * _rdz);

            // Clamp indices to ensure they fall within the total number of lattice points (0 to Px/Py/Pz - 1)
            ic = std::min(std::max(0, ic), _px - 1);
            jc = std::min(std::max(0, jc), _py - 1);
            kc = std::min(std::max(0, kc), _pz - 1);
        }

        friend std::ostream& operator<<(std::ostream& os, CartesianGrid3D& grid)
        {
            os << "Grid: {\n"
                << "\t{ minx: " << grid._min_x << " , min_y: " << grid._min_y << " , min_z: " << grid._min_z << " }\n"
                << "\t{ max_x: " << grid._max_x << " , max_y: " << grid._max_y << " , max_z: " << grid._max_z << " }\n"
                << "\t{ dx: " << grid._dx  << ", dy: " << grid._dy << ", dz: " << grid._dz << " }\n"
                << "\t{ nx: " << grid._nx  << ", ny: " << grid._ny << ", nz: " << grid._nz << " }\n"
                << "\t{ total number of cells: " << grid.totalNumCells() << " }\n"
                << "\t{ total number of lattice points: " << grid.totalNumPoints() << " }\n"
                << "}\n";
                return os;
        }
};

} // end namespace DGGML

#endif
