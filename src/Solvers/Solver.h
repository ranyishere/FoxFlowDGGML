#ifndef DGGML_SOLVER_H
#define DGGML_SOLVER_H

#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_ls.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif
//Added to fix depracated functionality in new version of sundials
typedef double realtype;
#define RCONST(x) x
#include <vector>
#include <variant>
#include <map>
#include <type_traits>
#include <algorithm>

#include "SundialsUtils.hpp"
#include "HelperFunctions.hpp"
#include "FemOperatorGroup.hpp"

#include <stdio.h>

// TODO: Need to generalize the sovler code and we need to add in the root finding fix from
// CajeteCMA repo
namespace DGGML {

    //TODO: Fix me!!! deprecated after version 7.0 of SUNDIALS, but will work with 6.7
    // Custom error handler function
    void customErrHandler(int error_code, const char *module, const char *function,
                          char *msg, void *eh_data) {
        // Implement logic to handle or ignore errors
        // For example, do nothing to effectively suppress the message
    }

    // Lets a model opt in to a spatial FEM operator for a given geocell. A model that defines
    //     using fem_operator_type = MyOperator;
    //     fem_operator_type* fem_operator(std::size_t geocell, bool has_solving_rules);
    // gets that operator (or nullptr) attached to the solver created by approximate_ssa for that
    // geocell. Models without these members get no FEM operator.
    template <typename M, typename = void>
    struct fem_operator_traits {
        using type = void*;
        static type get(M&, std::size_t, bool) { return nullptr; }
    };

    template <typename M>
    struct fem_operator_traits<M, std::void_t<typename M::fem_operator_type>> {
        using type = typename M::fem_operator_type*;
        static type get(M& m, std::size_t k, bool has_rules) { return m.fem_operator(k, has_rules); }
    };

    // An FEM operator that also provides
    //     void PrecSetup(double gamma);
    //     void PrecSolve(const double* r, double* z, double gamma);
    // is treated as a LINEAR, stiff term and integrated implicitly (ARKStep IMEX, matrix-free GMRES):
    //   Jacobian-vector product = Mult applied to the vector (Mult must be linear in y),
    //   PrecSetup(gamma) / PrecSolve(r, z, gamma): z = (I - gamma*J)^-1 r on the FEM segment
    //   (r and z point at the operator's own ndofs values; the rest of the state is untouched).
    // Operators without these members are integrated explicitly together with the grammar ODEs.
    template <typename T, typename = void>
    struct fem_operator_is_implicit : std::false_type {};

    template <typename T>
    struct fem_operator_is_implicit<T, std::void_t<decltype(
        std::declval<std::remove_pointer_t<T>&>().PrecSolve(std::declval<const double*>(),
                                                            std::declval<double*>(), 0.0))>>
        : std::true_type {};

    // FemOperatorType is an optional pointer to a spatial FEM operator (e.g. MFEM). When it is not
    // void*, the operator must provide: ndofs, fem_offset, Mult(t, y, ydot), Load(y) and Sync(y).
    //   Load(y): copy the operator's persistent field into y at fem_offset (called on construction).
    //   Sync(y): copy the FEM segment of y back to the persistent field (copy_back and destructor).
    // The solver is rebuilt for every SSA window, so the field must live in the operator.
    // The state vector layout is [grammar vars | FEM dofs | tau]; the solver sets fem_offset.
    // To host several operators in one solver, wrap them in DGGML::FemOperatorGroup<Op0, Op1, ...>
    // (see FemOperatorGroup.hpp): the group is itself a single operator whose segment is the
    // concatenation of its members' segments.
    template <typename ModelType, typename AnalyzedGrammarType, typename RuleMatchType, typename ComponentMatchType,
              typename FemOperatorType = void*>
    struct Solver
    {
        int flag; //generic reusable flag

        //Create the suncontext
        SUNContext ctx;
        void* arkode_mem;
        SUNLinearSolver LS = nullptr;   // matrix-free GMRES for the implicit FEM term (if any)

        static constexpr bool imex_capable = fem_operator_is_implicit<FemOperatorType>::value;
        bool use_imex = false;          // FEM term integrated implicitly (imex_capable && operator attached)

        realtype t_start, t_final, t, tout, dt_out;

        realtype reltol;
        realtype abstol;

        static constexpr int num_roots = 1;
        int roots_found[num_roots];
        int root_flag;

        sunindextype num_eq;

        N_Vector y;
        std::size_t global_nstep;
        std::size_t global_nfe;
        double last_prop_seen = -1.0;   // propensity used for the cached FSAL RHS
        //Note: by hiding it in the solver, we can add extra
        //data generated by the constructor and not the user
        // to be passed into the rhs
        struct UserDataType
        {
            ModelType& model;
            AnalyzedGrammarType& grammar_analysis;
            std::vector<std::size_t> solving_rules;
            // Local snapshot of rule_matches for the keys in solving_rules.
            // This is a VALUE (not a reference) so concurrent rehash of the
            // global rule_matches map cannot corrupt this thread's reads.
            typename RuleMatchType::container_type rule_matches;
            ComponentMatchType& component_matches;
            std::size_t& k; //TODO: figure out why I'm passing a reference
            double& geocell_propensity;
            double& tau;
            double waiting_time;
            std::unordered_map<double*, std::size_t> varmap;

            // Optional FEM operator for mesh-level PDE interaction
            FemOperatorType fem_op;

            UserDataType(ModelType& model, AnalyzedGrammarType& grammar_analysis, std::vector<std::size_t>& solving_rules,
                         const typename RuleMatchType::container_type& rule_matches_snapshot,
                         ComponentMatchType& component_matches, std::size_t& k,
                         double& geocell_propensity, double& tau, double waiting_time,
                         FemOperatorType fem_op = nullptr)
                         : model(model), grammar_analysis(grammar_analysis), solving_rules(solving_rules),
                         rule_matches(rule_matches_snapshot), component_matches(component_matches),
                         k(k), geocell_propensity(geocell_propensity), tau(tau), waiting_time(waiting_time),
                         fem_op(fem_op) {}
        };

        UserDataType user_data; //TODO: should this be a pointer?

        // Number of FEM degrees of freedom carried in the state vector (0 when no FEM operator)
        sunindextype fem_dofs() const
        {
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (user_data.fem_op) return static_cast<sunindextype>(user_data.fem_op->ndofs);
            }
            return 0;
        }

        //TODO: remember, the varset/varmap is only safe so long as the map doesn't change by another thread
        Solver(ModelType& model, AnalyzedGrammarType& grammar_analysis, std::vector<std::size_t> solving_rules,
               RuleMatchType& rule_matches,
               const typename RuleMatchType::container_type& rule_matches_snapshot,
               ComponentMatchType& component_matches, std::size_t& k, double& geocell_propensity,
               double& tau, double waiting_time, sunindextype num_eq, FemOperatorType fem_op = nullptr)
                : user_data(model, grammar_analysis, solving_rules, rule_matches_snapshot, component_matches, k,
                            geocell_propensity, tau, waiting_time, fem_op), num_eq(num_eq)
        {
            global_nstep = 0;
            global_nfe = 0;
            t_start = RCONST(0.0);
            dt_out = RCONST(user_data.model->settings.DELTA_T_MIN);
            t = t_start;
            tout = t_start + dt_out;
            t_final = RCONST(user_data.model->settings.DELTA);

            flag = SUNContext_Create(NULL, &ctx);

//            if(!DGGML::SundialsUtils::check_flag(&flag, "SUNContext_Create", 1))
//                std::cout << "Passed the error check, suncontext created\n";
//            std::cout << "*****NumEQ="<<num_eq<<"*****\n";


            std::set<double*> varset;
            //set the initial conditions
            for(auto& item : user_data.solving_rules)
            {
                auto& inst = user_data.rule_matches.at(item);
                auto& name = inst.name;
                auto& ode = user_data.grammar_analysis.solving_rules.find(name)->second;
                //TODO: find a more efficient way
                //need to construct a graph from the match keys and
                //the mapping from the orignal LHS to the match
                //auto induced_graph = induce_from_set(inst, user_data.component_matches, user_data.model->system_graph);
                std::map<std::size_t, std::size_t> lhs_vertex_map;
                construct_grammar_match_map(inst, user_data.grammar_analysis, lhs_vertex_map, user_data.component_matches);
//                std::cout << "for rule " << item << " we have a map: ";
//                for(auto& [k, v] : lhs_vertex_map)
//                    std::cout << "{ " << k << " -> " << v << " } ";
//                std::cout << "\n";
                ode.ic(user_data.model->system_graph, lhs_vertex_map, varset);
            }

            // grammar vars + FEM dofs + 1 for tau
            num_eq = static_cast<sunindextype>(varset.size()) + fem_dofs() + 1;

            y = NULL;
            y = N_VNew_Serial(num_eq, ctx);
            N_VConst(0.0, y);

            NV_Ith_S(y, num_eq - 1) = user_data.tau;
//            std::cout << "size of varset: " << varset.size() << "\n";
            std::size_t idx = 0;
            for(auto item : varset)
            {
                user_data.varmap.insert({item, idx});
                NV_Ith_S(y, idx) = *item;
                idx++;
            }
            varset.clear();

            // FEM segment sits right after the grammar variables; start from the operator's field
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (user_data.fem_op) {
                    user_data.fem_op->fem_offset = static_cast<int>(idx);
                    user_data.fem_op->Load(y);
                }
            }

            // === ODE SOLVER DUMP: one-time per-geocell snapshot ===
            // Set ODE_DUMP=1 to enable
            {
                static bool dump_enabled = (std::getenv("ODE_DUMP") != nullptr);
                if (dump_enabled && user_data.solving_rules.size() > 0) {
                    static int solver_id = 0;
                    solver_id++;
                    std::cout << "\n╔══════════════════════════════════════════════════╗" << std::endl;
                    std::cout << "║  ODE SOLVER #" << solver_id << "  (geocell " << user_data.k << ")  ║" << std::endl;
                    std::cout << "╚══════════════════════════════════════════════════╝" << std::endl;
                    std::cout << "  Equations: " << (num_eq) << " ODE vars + 1 tau" << std::endl;
                    std::cout << "  dt_out=" << dt_out << "  t_final=" << t_final << std::endl;

                    // Print variable map: which node attribute is each y[i]
                    std::cout << "\n  ── Variables (initial values) ──" << std::endl;

                    // Build reverse map: idx -> ptr
                    std::map<std::size_t, double*> rev;
                    for (auto& [ptr, idx2] : user_data.varmap) rev[idx2] = ptr;
                    for (auto& [idx2, ptr] : rev) {
                        std::cout << "    y[" << idx2 << "] = " << NV_Ith_S(y, idx2);
                        // Try to identify which node/attribute this pointer belongs to
                        for (auto it = user_data.model->system_graph.node_list_begin();
                             it != user_data.model->system_graph.node_list_end(); ++it) {
                            auto& nd = it->second.getData();
                            std::visit([&](auto& alt) {
                                using T = std::decay_t<decltype(alt)>;
                                // Check if ptr falls within this struct's memory
                                auto* base = reinterpret_cast<const char*>(&alt);
                                auto* p = reinterpret_cast<const char*>(ptr);
                                if (p >= base && p < base + sizeof(T)) {
                                    std::cout << "  ← node " << it->first
                                              << " (type_idx=" << nd.data.index() << ")";
                                }
                            }, nd.data);
                        }
                        std::cout << std::endl;
                    }

                    // Print rule matches
                    std::cout << "\n  ── Rule matches ──" << std::endl;
                    for (auto& item : user_data.solving_rules) {
                        auto& inst = user_data.rule_matches[item];
                        auto& name = inst.name;
                        std::map<std::size_t, std::size_t> lvm;
                        construct_grammar_match_map(inst, user_data.grammar_analysis,
                                                    lvm, user_data.component_matches);
                        std::cout << "    " << name << " {";
                        for (auto& [k2, v] : lvm)
                            std::cout << k2 << "->" << v << " ";
                        std::cout << "}" << std::endl;
                    }
                    std::cout << std::endl;
                }
            }

            if constexpr (imex_capable)
                use_imex = (user_data.fem_op != nullptr);

            // explicit part: grammar ODEs + tau (+ the FEM term when it is not implicit)
            // implicit part: the stiff FEM term, when the operator supports it
            arkode_mem = ARKStepCreate(rhs_exp, use_imex ? rhs_imp : NULL, t_start, y, ctx);

            reltol = 1.0e-6;
            abstol = 1.0e-8;

            flag = ARKStepSStolerances(arkode_mem, reltol, abstol);

            //set optional inputs
            flag = ARKStepSetUserData(arkode_mem, &user_data);

            // specify the root finding function having num_roots
            flag = ARKStepRootInit(arkode_mem, num_roots, root_func);

            ARKStepSetMaxStep(arkode_mem, dt_out);
            ARKStepSetMaxNumSteps(arkode_mem, 50000);  // allow many small steps for stiff transients
            attach_linear_solver();
//            std::cout << "Created solver\n";
            //set custom error handler for the solver


            // TODO: Comment back in 
            // ARKStepSetErrHandlerFn(arkode_mem, customErrHandler, NULL);
        }

        // (Re)create the matrix-free GMRES solver for the implicit FEM term and attach it
        void attach_linear_solver()
        {
            if constexpr (imex_capable) {
                if (!use_imex) return;
                SUNLinearSolver newLS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, 20, ctx);
                ARKStepSetLinearSolver(arkode_mem, newLS, NULL);   // NULL matrix: matrix-free
                ARKStepSetJacTimes(arkode_mem, NULL, jac_times);
                ARKStepSetPreconditioner(arkode_mem, prec_setup, prec_solve);
                ARKStepSetLinear(arkode_mem, 0);                   // implicit term is linear in y
                ARKStepSetEpsLin(arkode_mem, 1.0e-3);              // tight linear solves
                // fi(z) is not re-evaluated at the stage solutions: its stiff part would
                // amplify the linear-solve error (grid-scale noise); deduce it from the stage instead
                ARKStepSetDeduceImplicitRhs(arkode_mem, SUNTRUE);
                if (LS) SUNLinSolFree(LS);
                LS = newLS;
            }
        }

        // Restart the integrator from the current (t, y)
        void reinit_integrator()
        {
            ARKStepReInit(arkode_mem, rhs_exp, use_imex ? rhs_imp : NULL, t, y);
        }

        long int num_steps_taken() const
        {
            long int n = 0;
            ARKStepGetNumSteps(arkode_mem, &n);
            return n;
        }

        long int num_rhs_evals() const
        {
            long int ne = 0, ni = 0;
            ARKStepGetNumRhsEvals(arkode_mem, &ne, &ni);
            return ne + ni;
        }

        // Advance the integrator one reporting step. An optional t_cap (physical
        // time) makes the solver stop EARLY at t_cap instead of the next reporting
        // time tout -- used by the EveryRule time-queue so a scheduled clock event
        // fires at its exact time. When we stop at the cap (t_cap < tout) the
        // reporting cadence tout is left untouched so it resumes afterwards.
        void step(realtype t_cap = RCONST(-1.0))
        {

            // std::cout << "stepping solver...\n";
            if(num_eq == 0) return;

            // ARKODE caches f(t,y) from the previous step (FSAL). The tau
            // equation's rate is geocell_propensity, which the SSA recomputes
            // between steps (after EveryRule fires / events), so a stale cached
            // k1 makes the error test fail repeatedly (h -> hmin, flag -3).
            // Re-seed the integrator when the propensity changed.
            if(user_data.geocell_propensity != last_prop_seen)
            {
                reinit_integrator();
                last_prop_seen = user_data.geocell_propensity;
            }

            realtype target = tout;
            bool capped = false;
            if(t_cap >= RCONST(0.0) && t_cap < target)
            {
                target = t_cap;
                capped = true;
            }

            // Explicit Runga-Kutta Evolve to solve to the next output time (target)
            // Step to the target time value
            flag = ARKStepEvolve(arkode_mem, target, y, &t, ARK_NORMAL);
            // std::cout << "t: " << t << ", tout: " << tout << " flag: " << flag << "\n";
            // Check if a root was found (i.e., if the tau ODE crossed the waiting time threshold)
            if(flag == ARK_ROOT_RETURN)
            {
                root_flag = ARKStepGetRootInfo(arkode_mem, roots_found);
            }
            // successful solve
            else if(flag >= 0)
            {
                // Only advance the reporting cadence when we actually reached tout,
                // not when we stopped early at an EveryRule boundary cap.
                if(!capped || t >= tout - RCONST(1e-12))
                {
                    tout += dt_out;
                    tout = (tout > t_final) ? t_final : tout;
                }

            } else {
                std::cout << "Error in ARKStepEvolve: flag = " << flag << "\n";

                // Solver statistics
                long int nst, nst_a, nfe, netf;
                ARKStepGetNumSteps(arkode_mem, &nst);
                ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
                nfe = num_rhs_evals();
                ARKStepGetNumErrTestFails(arkode_mem, &netf);
                std::cout << "  Steps: " << nst << " (attempted: " << nst_a << ")\n";
                std::cout << "  RHS evals: " << nfe << "\n";
                std::cout << "  Error test failures: " << netf << "\n";

                // Step sizes
                realtype hcur, hlast;
                ARKStepGetCurrentStep(arkode_mem, &hcur);
                ARKStepGetLastStep(arkode_mem, &hlast);
                std::cout << "  Current step size: " << hcur << "\n";
                std::cout << "  Last step size: " << hlast << "\n";

                // Current y vector
                auto len = N_VGetLength(y);
                std::cout << "  num_eq: " << num_eq << "\n";
                std::cout << "  y vector (" << len << " elements):\n";
                for (int i = 0; i < len; i++)
                    std::cout << "    y[" << i << "] = " << NV_Ith_S(y, i) << "\n";

                // Varmap
                std::cout << "  varmap (" << user_data.varmap.size() << " entries):\n";
                for (auto& [ptr, idx] : user_data.varmap)
                    std::cout << "    ptr=" << ptr << " -> idx=" << idx << " (*ptr=" << *ptr << ")\n";

                // Call rhs manually to see current derivatives
                N_Vector ydot_diag = N_VNew_Serial(len, ctx);
                rhs(t, y, ydot_diag, &user_data);
                std::cout << "  ydot at failure:\n";
                for (int i = 0; i < len; i++)
                    std::cout << "    ydot[" << i << "] = " << NV_Ith_S(ydot_diag, i) << "\n";
                N_VDestroy(ydot_diag);

                // Check for NaN/Inf in y
                for (int i = 0; i < len; i++) {
                    double val = NV_Ith_S(y, i);
                    if (std::isnan(val)) std::cout << "  WARNING: y[" << i << "] is NaN!\n";
                    if (std::isinf(val)) std::cout << "  WARNING: y[" << i << "] is Inf!\n";
                }

                exit(1);
            }

        }

        static int root_func(realtype t, N_Vector y, realtype* gout, void* user_data)
        {
            UserDataType* local_ptr = (UserDataType *)user_data;
            auto last = N_VGetLength(y) - 1;

            //check for when the tau equation crosses the sample
            gout[0] = NV_Ith_S(y, last) - local_ptr->waiting_time;

            return 0;
        }

        // Full RHS (grammar ODEs + tau + FEM term); used for diagnostics and as the base of rhs_exp
        static int rhs(realtype t, N_Vector y, N_Vector ydot, void* user_data)
        {
            return rhs_body(t, y, ydot, user_data, true);
        }

        // Explicit part. The FEM term is evaluated here too so that grammar rules still see the
        // FEM derivatives in ydot (rules may mirror them into graph variables), but when the FEM
        // term is integrated implicitly its own segment is cleared before returning, so it is
        // not counted twice.
        static int rhs_exp(realtype t, N_Vector y, N_Vector ydot, void* user_data)
        {
            int ret = rhs_body(t, y, ydot, user_data, true);
            if constexpr (imex_capable) {
                UserDataType* p = (UserDataType *)user_data;
                if (p->fem_op) {
                    double* seg = N_VGetArrayPointer(ydot) + p->fem_op->fem_offset;
                    std::fill(seg, seg + p->fem_op->ndofs, 0.0);
                }
            }
            return ret;
        }

        // Implicit part: the (linear) FEM term only
        static int rhs_imp(realtype t, N_Vector y, N_Vector ydot, void* user_data)
        {
            N_VConst(0.0, ydot);
            if constexpr (imex_capable) {
                UserDataType* p = (UserDataType *)user_data;
                if (p->fem_op) p->fem_op->Mult(t, y, ydot);
            }
            return 0;
        }

        // Jv for the implicit term: it is linear, so J*v is the term applied to v
        static int jac_times(N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy,
                             void* user_data, N_Vector tmp)
        {
            return rhs_imp(t, v, Jv, user_data);
        }

        static int prec_setup(realtype t, N_Vector y, N_Vector fy, booleantype jok,
                              booleantype* jcurPtr, realtype gamma, void* user_data)
        {
            if constexpr (imex_capable) {
                UserDataType* p = (UserDataType *)user_data;
                if (p->fem_op) p->fem_op->PrecSetup(gamma);
            }
            *jcurPtr = SUNTRUE;
            return 0;
        }

        static int prec_solve(realtype t, N_Vector y, N_Vector fy, N_Vector r, N_Vector z,
                              realtype gamma, realtype delta, int lr, void* user_data)
        {
            N_VScale(1.0, r, z);   // identity outside the FEM segment
            if constexpr (imex_capable) {
                UserDataType* p = (UserDataType *)user_data;
                if (p->fem_op) {
                    const int off = p->fem_op->fem_offset;
                    p->fem_op->PrecSolve(N_VGetArrayPointer(r) + off, N_VGetArrayPointer(z) + off, gamma);
                }
            }
            return 0;
        }

        static int rhs_body(realtype t, N_Vector y, N_Vector ydot, void* user_data, bool with_fem)
        {
            UserDataType* local_ptr = (UserDataType *)user_data;
            //fill ydot with zeros
            auto global_length = N_VGetLength(ydot);
            for(auto i = 0; i < global_length; i++)
                NV_Ith_S(ydot, i) = 0.0;

            // Spatial FEM PDE terms (if an FEM operator is attached); operates directly on
            // the SUNDIALS array pointers at the operator's offset
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (with_fem && local_ptr->fem_op)
                    local_ptr->fem_op->Mult(t, y, ydot);
            }

            // === ODE DIAGNOSTIC: print full system on first call with actual equations ===
            // Set ODE_DIAG_ENABLED=1 env var to enable
            static int ode_diag_count = 0;
            static bool ode_diag_enabled = (std::getenv("ODE_DIAG_ENABLED") != nullptr);
            bool do_diag = ode_diag_enabled && (ode_diag_count < 20) && (local_ptr->solving_rules.size() > 0);
            if (do_diag) {
                ode_diag_count++;
                std::cout << "\n=== ODE SYSTEM DIAGNOSTIC (RHS call #" << ode_diag_count << ", geocell=" << local_ptr->k << ", t=" << t << ") ===" << std::endl;
                std::cout << "Total ODE variables: " << global_length << std::endl;
                std::cout << "Variable map (ptr -> index, current value):" << std::endl;
                for (auto& [ptr, idx] : local_ptr->varmap)
                    std::cout << "  y[" << idx << "] = " << *ptr << std::endl;
                std::cout << "\nSolving rule matches: " << local_ptr->solving_rules.size() << std::endl;
            }

            for(auto& item : local_ptr->solving_rules)
            {
                auto& inst = local_ptr->rule_matches.at(item);
                auto& name = inst.name;
                auto& eq = local_ptr->grammar_analysis.solving_rules.find(name)->second;
                std::map<std::size_t, std::size_t> lhs_vertex_map;
                construct_grammar_match_map(inst, local_ptr->grammar_analysis, lhs_vertex_map, local_ptr->component_matches);

                if (do_diag) {
                    // snapshot ydot before this rule fires
                    std::vector<double> before(global_length);
		    std::cout << "global_length: " << global_length << "\n";
                    for (auto i = 0; i < global_length; i++) before[i] = NV_Ith_S(ydot, i);

                    eq.ode(local_ptr->model->system_graph, lhs_vertex_map, y, ydot, local_ptr->varmap);

                    std::cout << "  Rule '" << name << "' match {";
                    for (auto& [k,v] : lhs_vertex_map) std::cout << k << "->" << v << " ";
                    std::cout << "}: ";
                    bool any = false;
                    for (auto i = 0; i < global_length; i++) {
                        double delta = NV_Ith_S(ydot, i) - before[i];
                        if (std::abs(delta) > 1e-15) {
                            if (any) std::cout << ", ";
                            std::cout << "ydot[" << i << "]+=" << delta;
                            any = true;
                        }
                    }
                    if (!any) std::cout << "(no contribution)";
                    std::cout << std::endl;
                } else {
                    eq.ode(local_ptr->model->system_graph, lhs_vertex_map, y, ydot, local_ptr->varmap);
                }
            }

            if (do_diag) {
                std::cout << "\nFinal ydot:" << std::endl;
                for (auto i = 0; i < global_length; i++)
                    std::cout << "  ydot[" << i << "] = " << NV_Ith_S(ydot, i) << std::endl;
                std::cout << "=== END ODE DIAGNOSTIC ===\n" << std::endl;
            }

            NV_Ith_S(ydot, global_length-1) = local_ptr->geocell_propensity;

            local_ptr = nullptr;

            return 0;
        }

        // Copy of the FEM segment of y, so it survives a rebuild of the state vector
        std::vector<double> save_fem_state()
        {
            std::vector<double> saved;
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (user_data.fem_op) {
                    const double* src = N_VGetArrayPointer(y) + user_data.fem_op->fem_offset;
                    saved.assign(src, src + user_data.fem_op->ndofs);
                }
            }
            return saved;
        }

        // Place a saved FEM segment at its new offset (right after the grammar variables)
        void restore_fem_state(const std::vector<double>& saved, std::size_t new_offset)
        {
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (user_data.fem_op) {
                    user_data.fem_op->fem_offset = static_cast<int>(new_offset);
                    std::copy(saved.begin(), saved.end(), N_VGetArrayPointer(y) + new_offset);
                }
            }
        }

        void copy_back()
        {
            for(auto& [item, key] : user_data.varmap)
            {
                *item = NV_Ith_S(y, key);
            }
            auto global_length = N_VGetLength(y);
            user_data.tau = NV_Ith_S(y, global_length-1);

            // Update FEM fields (e.g. GridFunction) from the SUNDIALS vector
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (user_data.fem_op)
                    user_data.fem_op->Sync(y);
            }
        }

        // Refresh the continuous ODE state vector y from the CURRENT graph node
        // data WITHOUT resetting the integrator time t or the warped Gillespie
        // clock (tau). Used after an EveryRule bulk rewrite mutates node data
        // mid-window: the ODE state variables must follow the graph, but the
        // waiting-time accounting (t, tau, exp_sample) must be preserved so the
        // stochastic timeline is not corrupted. reinit() cannot be used here
        // because it zeroes tau. No-op when only the tau equation exists.
        void resync_state()
        {
            if(num_eq <= 1) return;   // no real ODE state variables to refresh
            for(auto& [ptr, idx] : user_data.varmap)
                NV_Ith_S(y, idx) = *ptr;
            reinit_integrator();
            if(tout <= t)
            {
                tout = t + dt_out;
                tout = (tout > t_final) ? t_final : tout;
            }
        }

        //TODO: make this function reset the whole y vector
        //since it may change in size
        void reinit(std::vector<std::size_t> solving_rules, const typename RuleMatchType::container_type& rm_snapshot)
        {
            long int nst = num_steps_taken(), nfe_local = num_rhs_evals();
            global_nstep += nst;
            global_nfe += nfe_local;

            user_data.solving_rules = solving_rules;
            user_data.rule_matches = rm_snapshot;

            auto pre_eq = num_eq;

            /*
            num_eq = 0;
            for(auto& item : user_data.solving_rules)
            {
                auto& name = user_data.rule_matches[item].name;
                num_eq += user_data.grammar_analysis.solving_rules.find(name)->second.num_eq;
            }
            num_eq++; //one extra equation for tau
            std::cout << "Previo # of eq: " << pre_eq << "\n";
            std::cout << "Reinit # of eq: " << num_eq << "\n";
            */

            user_data.varmap.clear();

            //rebuild the ic
            std::set<double*> varset;
            //set the initial conditions
            for(auto& item : user_data.solving_rules)
            {
                auto& inst = user_data.rule_matches.at(item);
                auto& name = inst.name;
                auto& ode = user_data.grammar_analysis.solving_rules.find(name)->second;
                //TODO: find a more efficient way
                std::map<std::size_t, std::size_t> lhs_vertex_map;
                construct_grammar_match_map(inst, user_data.grammar_analysis, lhs_vertex_map, user_data.component_matches);
                ode.ic(user_data.model->system_graph, lhs_vertex_map, varset);
            }

            num_eq = static_cast<sunindextype>(varset.size()) + fem_dofs() + 1;

            // NV_Ith_S(y, varset.size()) = user_data.tau;
            std::size_t idx = 0;
            // std::cout << "Global length: " << N_VGetLength(y) << "\n";
            // std::cout << "Local length: " << N_VGetLocalLength(y) << "\n";
            // std::cout << "Varset length: " << varset.size() << "\n";

            auto fem_saved = save_fem_state();
            N_VDestroy(y);
            y = N_VNew_Serial(num_eq, ctx);
            N_VConst(0.0, y);

            for(auto item : varset)
            {
                user_data.varmap.insert({item, idx});
                NV_Ith_S(y, idx) = *item;
                idx++;
            }
            restore_fem_state(fem_saved, varset.size());

            user_data.tau = 0.0;
            NV_Ith_S(y, num_eq - 1) = 0.0;

            varset.clear();

            if(num_eq == pre_eq)
                reinit_integrator();
            else
            {
                ARKStepResize(arkode_mem, y, 1.0, t, NULL, NULL);
                attach_linear_solver();
                reinit_integrator();
            }
            // Ensure tout is ahead of t after reinit so the next
            // ARKStepEvolve call has a valid forward target.
            if (tout <= t) {
                tout = t + dt_out;
                tout = (tout > t_final) ? t_final : tout;
            }

        }

        //TODO: make this function reset the whole y vector
        //since it may change in size
        void reinit(std::vector<std::size_t> solving_rules)
        {
            long int nst = num_steps_taken(), nfe_local = num_rhs_evals();
            global_nstep += nst;
            global_nfe += nfe_local;
            user_data.solving_rules = solving_rules;

            auto pre_eq = num_eq;

            /*
            num_eq = 0;
            for(auto& item : user_data.solving_rules)
            {
                auto& name = user_data.rule_matches[item].name;
                num_eq += user_data.grammar_analysis.solving_rules.find(name)->second.num_eq;
            }
            num_eq++; //one extra equation for tau
            std::cout << "Previo # of eq: " << pre_eq << "\n";
            std::cout << "Reinit # of eq: " << num_eq << "\n";
            */

            user_data.varmap.clear();

            //rebuild the ic
            std::set<double*> varset;
            //set the initial conditions
            for(auto& item : user_data.solving_rules)
            {
                auto& inst = user_data.rule_matches.at(item);
                auto& name = inst.name;
                auto& ode = user_data.grammar_analysis.solving_rules.find(name)->second;
                //TODO: find a more efficient way
                std::map<std::size_t, std::size_t> lhs_vertex_map;
                construct_grammar_match_map(inst, user_data.grammar_analysis, lhs_vertex_map, user_data.component_matches);
                ode.ic(user_data.model->system_graph, lhs_vertex_map, varset);
            }

            num_eq = static_cast<sunindextype>(varset.size()) + fem_dofs() + 1;

            // NV_Ith_S(y, varset.size()) = user_data.tau;
            std::size_t idx = 0;
            // std::cout << "Global length: " << N_VGetLength(y) << "\n";
            // std::cout << "Local length: " << N_VGetLocalLength(y) << "\n";
            // std::cout << "Varset length: " << varset.size() << "\n";

            auto fem_saved = save_fem_state();
            N_VDestroy(y);
            y = N_VNew_Serial(num_eq, ctx);
            N_VConst(0.0, y);

            for(auto item : varset)
            {
                user_data.varmap.insert({item, idx});
                NV_Ith_S(y, idx) = *item;
                idx++;
            }
            restore_fem_state(fem_saved, varset.size());


            user_data.tau = 0.0;
            NV_Ith_S(y, num_eq - 1) = 0.0;

            varset.clear();

            if(num_eq == pre_eq)
                reinit_integrator();
            else
            {
                ARKStepResize(arkode_mem, y, 1.0, t, NULL, NULL);
                attach_linear_solver();
                reinit_integrator();
            }
            // Ensure tout is ahead of t after reinit so the next
            // ARKStepEvolve call has a valid forward target.
            if (tout <= t) {
                tout = t + dt_out;
                tout = (tout > t_final) ? t_final : tout;
            }

        }

        void print_stats()
        {
            long int nst, nst_a, nfe, netf;
            /* Print some final statistics */
            flag = ARKStepGetNumSteps(arkode_mem, &nst);
            global_nstep += nst;
            flag = ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
            nfe = num_rhs_evals();
            flag = ARKStepGetNumErrTestFails(arkode_mem, &netf);

            printf("\nFinal Solver Statistics:\n");
            printf("   Internal solver steps = %li (attempted = %li)\n", nst, nst_a);
            printf("   Total RHS evals = %li\n", nfe);
            printf("   Total number of error test failures = %li\n\n", netf);
            std::cout << "Global number of steps: " << global_nstep << "\n";
        }
        ~Solver()
        {
            //std::cout << "Starting solver destruction\n";
            //std::cout << "Global length: " << N_VGetLength(y) << "\n";
            //std::cout << "Local length: " << N_VGetLocalLength(y) << "\n";
            //std::cout << "Varset length: " << user_data.varmap.size() << "\n";
            if constexpr (!std::is_same_v<FemOperatorType, void*>) {
                if (user_data.fem_op)
                    user_data.fem_op->Sync(y);   // persist the field before y is destroyed
            }
            N_VDestroy(y);
            //std::cout << "Destroyed y vector\n";
            ARKStepFree(&arkode_mem); //free the solver memory
            if (LS) SUNLinSolFree(LS);
            //std::cout << "Destroyed arkode_mem\n";
            SUNContext_Free(&ctx); //always call prior to MPI_Finalize
            //std::cout << "Destroyed the solver\n";

            /* Step 13: Finalilze MPI, if used */
        }
    };
} // DGGML

#endif //DGGML_SOLVER_H