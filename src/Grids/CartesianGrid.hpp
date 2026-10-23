#ifndef DGGML_CARTESIANGRID_HPP
#define DGGML_CARTESIANGRID_HPP

namespace DGGML {


    class CartesianGrid {
        public:
            CartesianGrid() {}

        int totalNumCells() const { return 0; }

        // 3D version
        virtual void locatePoint(const double xp, const double yp,
                const double zp, int& ic, int& jc, int& kc) const = 0;

        // 2D version
        virtual void locatePoint(const double xp, const double yp,
                int& ic, int& jc) const = 0;
    };
    
}
#endif //DGGML_CARTESIANGRID_HPP
