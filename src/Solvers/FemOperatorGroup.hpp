#ifndef DGGML_FEM_OPERATOR_GROUP_HPP
#define DGGML_FEM_OPERATOR_GROUP_HPP

#include <nvector/nvector_serial.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace DGGML {

    // Composite FEM operator: hosts several spatial operators (of possibly different types) in a
    // single geocell solver. The group satisfies the same contract as a single operator
    // (ndofs, fem_offset, Mult, Load, Sync), so Solver needs no knowledge of it. A model opts in with
    //
    //     using fem_operator_type = DGGML::FemOperatorGroup<HeatOp, ElasticityOp>;
    //     fem_operator_type* fem_operator(std::size_t k, bool has_rules)
    //     { return group->claim(k, has_rules) ? group.get() : nullptr; }
    //
    // The state vector segment is the concatenation of the member segments, in constructor order:
    //
    //     y = [ grammar variables | op0 dofs | op1 dofs | ... | tau ]
    //
    // Each member must provide, as in the single-operator case:
    //   int ndofs, int fem_offset, Mult(t, y, ydot), Load(y), Sync(y)
    // Mult must ACCUMULATE into ydot (+=) over its own segment; the solver zeroes ydot beforehand.
    // Before every Mult/Load/Sync the group writes each member's fem_offset, so members (and
    // grammar rules holding a member pointer) always see their current absolute offset.
    template <typename... Ops>
    class FemOperatorGroup {
        static_assert(sizeof...(Ops) > 0, "FemOperatorGroup needs at least one operator");

    public:
        int ndofs = 0;        // total dofs over all members
        int fem_offset = 0;   // set by the Solver; start of the group's segment in y

        explicit FemOperatorGroup(std::shared_ptr<Ops>... operators)
            : ops(std::move(operators)...)
        {
            std::apply([this](auto&... op) { ((ndofs += op->ndofs), ...); }, ops);
        }

        void Mult(double t, N_Vector y, N_Vector ydot)
        {
            for_each_member([&](auto& op) { op.Mult(t, y, ydot); });
        }

        void Load(N_Vector y)
        {
            for_each_member([&](auto& op) { op.Load(y); });
        }

        void Sync(N_Vector y)
        {
            for_each_member([&](auto& op) { op.Sync(y); });
        }

        // Access the I-th member, e.g. to read its persistent field.
        template <std::size_t I>
        auto& get() { return *std::get<I>(ops); }

        // Absolute offset in y of the I-th member's segment, given the current fem_offset.
        template <std::size_t I>
        int offset_of() const
        {
            int off = fem_offset;
            std::size_t i = 0;
            std::apply([&](const auto&... op) {
                ((i++ < I ? (void)(off += op->ndofs) : (void)0), ...);
            }, ops);
            return off;
        }

        // The group is integrated in exactly one geocell's solver: the first one that has solving
        // rule matches claims it (same policy as a single operator).
        bool claim(std::size_t k, bool has_rules)
        {
            std::lock_guard<std::mutex> lock(claim_mutex);
            if (owner == npos && has_rules) {
                owner = k;
                std::cout << "[FemOperatorGroup] hosted by geocell " << k << "\n";
            }
            return owner == k;
        }

    private:
        static constexpr std::size_t npos = static_cast<std::size_t>(-1);
        std::size_t owner = npos;
        std::mutex claim_mutex;

        std::tuple<std::shared_ptr<Ops>...> ops;

        template <typename F>
        void for_each_member(F&& f)
        {
            int off = fem_offset;
            auto visit = [&](auto& op) {
                op->fem_offset = off;
                f(*op);
                off += op->ndofs;
            };
            std::apply([&](auto&... op) { (visit(op), ...); }, ops);
        }
    };

} // DGGML

#endif //DGGML_FEM_OPERATOR_GROUP_HPP
