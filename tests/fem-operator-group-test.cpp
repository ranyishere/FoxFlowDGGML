#include <algorithm>
#include <memory>
#include <vector>

#include "catch.hpp"
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include "Solvers/FemOperatorGroup.hpp"

namespace {

    // Minimal operators following the single-operator contract: du/dt = rate * u
    struct ScaleOp {
        int ndofs;
        int fem_offset = 0;
        double rate;
        std::vector<double> u;

        ScaleOp(int n, double r, double init) : ndofs(n), rate(r), u(n, init) {}

        void Mult(double, N_Vector y, N_Vector ydot)
        {
            const double* yp = N_VGetArrayPointer(y) + fem_offset;
            double* dp = N_VGetArrayPointer(ydot) + fem_offset;
            for (int i = 0; i < ndofs; ++i) dp[i] += rate * yp[i];
        }
        void Load(N_Vector y) { std::copy(u.begin(), u.end(), N_VGetArrayPointer(y) + fem_offset); }
        void Sync(N_Vector y)
        {
            const double* s = N_VGetArrayPointer(y) + fem_offset;
            std::copy(s, s + ndofs, u.begin());
        }
    };

    // A different type, to check heterogeneous members: du/dt = -u (offset constant source)
    struct DecayOp {
        int ndofs;
        int fem_offset = 0;
        std::vector<double> u;

        DecayOp(int n, double init) : ndofs(n), u(n, init) {}

        void Mult(double, N_Vector y, N_Vector ydot)
        {
            const double* yp = N_VGetArrayPointer(y) + fem_offset;
            double* dp = N_VGetArrayPointer(ydot) + fem_offset;
            for (int i = 0; i < ndofs; ++i) dp[i] += -yp[i] + 1.0;
        }
        void Load(N_Vector y) { std::copy(u.begin(), u.end(), N_VGetArrayPointer(y) + fem_offset); }
        void Sync(N_Vector y)
        {
            const double* s = N_VGetArrayPointer(y) + fem_offset;
            std::copy(s, s + ndofs, u.begin());
        }
    };

}

TEST_CASE("FemOperatorGroup concatenates member segments", "[fem-operator-group]")
{
    auto a = std::make_shared<ScaleOp>(3, 2.0, 1.0);
    auto b = std::make_shared<DecayOp>(2, 5.0);
    DGGML::FemOperatorGroup<ScaleOp, DecayOp> group(a, b);

    REQUIRE(group.ndofs == 5);

    // layout: [2 grammar vars | a (3) | b (2) | tau]
    const int grammar = 2;
    group.fem_offset = grammar;
    REQUIRE(group.offset_of<0>() == 2);
    REQUIRE(group.offset_of<1>() == 5);

    SUNContext ctx;
    REQUIRE(SUNContext_Create(NULL, &ctx) == 0);
    const sunindextype n = grammar + group.ndofs + 1;
    N_Vector y = N_VNew_Serial(n, ctx);
    N_Vector ydot = N_VNew_Serial(n, ctx);
    N_VConst(0.0, y);
    N_VConst(0.0, ydot);

    SECTION("Load places each member at its own offset")
    {
        group.Load(y);
        REQUIRE(a->fem_offset == 2);
        REQUIRE(b->fem_offset == 5);
        for (int i = 0; i < 3; ++i) REQUIRE(NV_Ith_S(y, 2 + i) == 1.0);
        for (int i = 0; i < 2; ++i) REQUIRE(NV_Ith_S(y, 5 + i) == 5.0);
        REQUIRE(NV_Ith_S(y, 0) == 0.0);
        REQUIRE(NV_Ith_S(y, n - 1) == 0.0);
    }

    SECTION("Mult fills only the member segments")
    {
        group.Load(y);
        group.Mult(0.0, y, ydot);
        for (int i = 0; i < 3; ++i) REQUIRE(NV_Ith_S(ydot, 2 + i) == Approx(2.0));
        for (int i = 0; i < 2; ++i) REQUIRE(NV_Ith_S(ydot, 5 + i) == Approx(-4.0));
        REQUIRE(NV_Ith_S(ydot, 0) == 0.0);
        REQUIRE(NV_Ith_S(ydot, 1) == 0.0);
        REQUIRE(NV_Ith_S(ydot, n - 1) == 0.0);
    }

    SECTION("Sync copies each member's segment back after the offset moves")
    {
        group.Load(y);

        // simulate the solver rebuilding y with a different number of grammar variables
        N_Vector y2 = N_VNew_Serial(n + 3, ctx);
        N_VConst(0.0, y2);
        std::copy(N_VGetArrayPointer(y) + 2, N_VGetArrayPointer(y) + 2 + group.ndofs,
                  N_VGetArrayPointer(y2) + 5);
        group.fem_offset = 5;
        for (int i = 0; i < group.ndofs; ++i) NV_Ith_S(y2, 5 + i) += 10.0;

        group.Sync(y2);
        REQUIRE(a->fem_offset == 5);
        REQUIRE(b->fem_offset == 8);
        for (int i = 0; i < 3; ++i) REQUIRE(a->u[i] == Approx(11.0));
        for (int i = 0; i < 2; ++i) REQUIRE(b->u[i] == Approx(15.0));
        N_VDestroy(y2);
    }

    N_VDestroy(y);
    N_VDestroy(ydot);
    SUNContext_Free(&ctx);
}

TEST_CASE("FemOperatorGroup claim is first-come with rules", "[fem-operator-group]")
{
    DGGML::FemOperatorGroup<ScaleOp> group(std::make_shared<ScaleOp>(1, 1.0, 0.0));

    REQUIRE_FALSE(group.claim(3, false));   // no rules: does not claim
    REQUIRE(group.claim(4, true));          // first with rules claims
    REQUIRE_FALSE(group.claim(5, true));    // already owned
    REQUIRE(group.claim(4, true));          // owner keeps it
}
