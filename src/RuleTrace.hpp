#ifndef DGGML_RULE_TRACE_HPP
#define DGGML_RULE_TRACE_HPP

// Rule-fire trace hook.
//
// Records, for every fire of a selected rule, the state of the matched nodes
// before and after the rule's update, plus their 1-hop neighbourhood, the
// edge changes, the propensity at fire time and any where-clause scalar
// locals the generated code reports through Trace::local().
//
// Controlled by the FoxFlow source (generated code calls begin_stage() for each
// simulation stage):
//   trace(simulationName, "rule1", "rule2")   or "*" for all rules
// and/or by environment variables (everything is off when neither is given):
//   FOXFLOW_TRACE        comma separated rule names, or "*" for all rules
//   FOXFLOW_TRACE_FILE   output path (overrides the per-simulation file chosen
//                        by trace(...); default RuleTrace.jsonl, truncated at start)
//   FOXFLOW_TRACE_MAX    stop after this many records (default unlimited)
//
// One JSON object per line. The (sim_step, ssa_step, rule) triple matches the
// rows written by RuleFireListener.csv.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if __has_include(<torch/torch.h>)
#include <torch/torch.h>
#define DGGML_TRACE_HAS_TORCH 1
#endif

namespace DGGML { namespace Trace {

    struct State {
        std::set<std::string> env_rules;  // from FOXFLOW_TRACE
        bool env_all = false;
        bool has_env_file = false;
        std::set<std::string> rules;      // effective: environment + trace(...) of the current stage
        bool all = false;
        long long max_records = -1;
        long long count = 0;
        std::string file = "RuleTrace.jsonl";
        std::ofstream out;
        std::mutex mu;
    };

    inline State& state() { static State s; return s; }

    inline void configure() {
        static const bool done = []() {
            auto& s = state();
            if (const char* r = std::getenv("FOXFLOW_TRACE")) {
                std::stringstream ss(r);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
                    if (tok == "*") s.env_all = true;
                    else if (!tok.empty()) s.env_rules.insert(tok);
                }
            }
            if (const char* f = std::getenv("FOXFLOW_TRACE_FILE")) { s.file = f; s.has_env_file = true; }
            if (const char* m = std::getenv("FOXFLOW_TRACE_MAX")) s.max_records = std::atoll(m);
            s.rules = s.env_rules;
            s.all = s.env_all;
            if (s.all || !s.rules.empty()) {
                s.out.open(s.file, std::ios::out | std::ios::trunc);
                std::cout << "[FoxFlow trace] writing rule fires to " << s.file << "\n";
            }
            return true;
        }();
        (void)done;
    }

    // Called by generated code at the start of each simulation stage with the rules named in
    // trace(simulation, "rule", ...) (empty when the stage has no trace statement) and the
    // file to write them to. Environment settings stay in effect and take precedence for the file.
    inline void begin_stage(const std::vector<std::string>& dsl_rules, const std::string& dsl_file) {
        configure();
        auto& s = state();
        std::lock_guard<std::mutex> lk(s.mu);
        s.rules = s.env_rules;
        s.all = s.env_all;
        for (const auto& r : dsl_rules) {
            if (r == "*") s.all = true;
            else s.rules.insert(r);
        }
        const bool want = s.all || !s.rules.empty();
        const std::string file = s.has_env_file ? s.file : dsl_file;
        if (!want) {
            if (s.out.is_open()) s.out.close();
            return;
        }
        if (!s.out.is_open() || s.file != file) {
            if (s.out.is_open()) s.out.close();
            s.file = file;
            s.count = 0;
            s.out.open(s.file, std::ios::out | std::ios::trunc);
            std::cout << "[FoxFlow trace] writing rule fires to " << s.file << "\n";
        }
    }

    inline bool enabled(const std::string& rule) {
        configure();
        auto& s = state();
        if (!s.out.is_open()) return false;
        if (s.max_records >= 0 && s.count >= s.max_records) return false;
        return s.all || s.rules.count(rule) > 0;
    }

    // Per-thread context: step counters set by the liveliness monitor right
    // before a rule fires, and where-locals collected while the update runs.
    inline thread_local int  sim_step = -1;
    inline thread_local int  ssa_step = -1;
    inline thread_local bool active   = false;
    inline thread_local std::vector<std::pair<std::string, std::string>> locals;  // name -> JSON value

    inline void set_step(int sim, int ssa) { sim_step = sim; ssa_step = ssa; }

    // ---- JSON helpers -----------------------------------------------------

    inline void jnum(std::ostream& os, double v) {
        if (std::isnan(v))      os << "\"nan\"";
        else if (std::isinf(v)) os << (v > 0 ? "\"inf\"" : "\"-inf\"");
        else                    os << std::setprecision(17) << v;
    }

    template <class T, class = void> struct has_numel : std::false_type {};
    template <class T>
    struct has_numel<T, std::void_t<decltype(std::declval<const T&>().numel())>> : std::true_type {};

    template <class T, class = void> struct has_field_names : std::false_type {};
    template <class T>
    struct has_field_names<T, std::void_t<decltype(T::field_names())>> : std::true_type {};

    // Minimal cereal-like archive: serialize(archive) calls archive(field...)
    // in declaration order; we print each field as "name": value.
    struct DumpArchive {
        std::ostringstream os;
        std::vector<std::string> names;
        std::size_t idx = 0;
        bool skip_first = false;

        template <class T, class... R>
        DumpArchive& operator()(T& v, R&... rest) {
            one(v);
            if constexpr (sizeof...(R) > 0) (*this)(rest...);
            return *this;
        }

        template <class T>
        void one(T& v) {
            std::string name = idx < names.size() ? names[idx] : "a" + std::to_string(idx);
            const bool first_attr = (idx == 0);
            ++idx;
            // The first attribute is the spatial Position: a stale mirror of
            // node.position, which is the authoritative value. Skip it for
            // nodes (local values are written with write() and unaffected).
            if (first_attr && skip_first) return;
            if (os.tellp() > 0) os << ",";
            os << "\"" << name << "\":";
            write(v);
        }

        template <class T>
        void write(const T& v) {
            if constexpr (std::is_arithmetic_v<T>) {
                jnum(os, static_cast<double>(v));
            }
#ifdef DGGML_TRACE_HAS_TORCH
            else if constexpr (has_numel<T>::value) {
                auto f = v.detach().to(torch::kFloat64).contiguous().flatten();
                const double* p = f.template data_ptr<double>();
                os << "[";
                for (int64_t i = 0; i < f.numel(); ++i) {
                    if (i) os << ",";
                    jnum(os, p[i]);
                }
                os << "]";
            }
#endif
            else {
                os << "\"<unsupported>\"";
            }
        }
    };

    // Called from generated where-clauses after each local definition.
    template <class T>
    inline void local(const char* name, const T& v) {
        if (!active) return;
        DumpArchive ar;
        ar.write(v);
        locals.emplace_back(name, ar.os.str());
    }

    template <class NodeData>
    inline std::string node_json(std::size_t id, NodeData& d) {
        std::ostringstream os;
        os << "{\"id\":" << id << ",\"type\":" << d.data.index() << ",\"pos\":[";
        for (int i = 0; i < 3; ++i) { if (i) os << ","; jnum(os, d.position[i]); }
        os << "],\"data\":{";
        std::visit([&](auto& alt) {
            using A = std::decay_t<decltype(alt)>;
            DumpArchive ar;
            ar.skip_first = true;
            if constexpr (has_field_names<A>::value) {
                for (auto& n : A::field_names()) ar.names.push_back(n);
            }
            alt.serialize(ar);
            os << ar.os.str();
        }, d.data);
        os << "}}";
        return os.str();
    }

    using Edge = std::pair<std::size_t, std::size_t>;
    inline Edge norm(std::size_t a, std::size_t b) { return a < b ? Edge{a, b} : Edge{b, a}; }

    // State captured before the rule's update runs. Tensors inside node data
    // are shared between the LHS snapshot and the working copy, so the "before"
    // view must be serialized eagerly.
    struct Snapshot {
        std::string rule;
        std::vector<std::string> before;
        std::vector<std::string> context;
        std::set<std::size_t> before_ids;
        std::set<Edge> edges_before;
        double propensity = 0.0;
        std::vector<std::pair<std::size_t, std::size_t>> match;  // pattern key -> system id
    };

    template <class Graph, class Map, class Propensity>
    inline Snapshot capture_before(const std::string& rule, Graph& system_graph, Graph& lhs_match,
                                   Map& lhs_vertex_map, Propensity&& propensity_fn) {
        Snapshot s;
        s.rule = rule;
        for (auto& [k, v] : lhs_vertex_map) s.match.emplace_back(k, v);
        for (auto& [k, node] : lhs_match.getNodeSetRef()) {
            s.before_ids.insert(k);
            s.before.push_back(node_json(k, node.getData()));
        }
        std::set<std::size_t> ctx;
        for (auto id : s.before_ids) {
            auto it = system_graph.findNode(id);
            if (it == system_graph.node_list_end()) continue;
            for (auto n : system_graph.out_neighbors(id)) {
                s.edges_before.insert(norm(id, n));
                if (!s.before_ids.count(n)) ctx.insert(n);
            }
            for (auto n : system_graph.in_neighbors(id)) {
                s.edges_before.insert(norm(n, id));
                if (!s.before_ids.count(n)) ctx.insert(n);
            }
        }
        for (auto id : ctx) {
            auto it = system_graph.findNode(id);
            if (it == system_graph.node_list_end()) continue;
            s.context.push_back(node_json(id, it->second.getData()));
        }
        try { s.propensity = propensity_fn(); } catch (...) { s.propensity = std::nan(""); }
        locals.clear();
        active = true;
        return s;
    }

    template <class Graph, class Updates>
    inline void write_after(Snapshot& s, Graph& lhs_match_copy, Updates& changes) {
        active = false;
        std::ostringstream os;
        std::set<std::size_t> after_ids;
        std::vector<std::string> after;
        for (auto& [k, node] : lhs_match_copy.getNodeSetRef()) {
            after_ids.insert(k);
            after.push_back(node_json(k, node.getData()));
        }
        std::vector<std::size_t> created, destroyed;
        for (auto id : after_ids) if (!s.before_ids.count(id)) created.push_back(id);
        for (auto id : s.before_ids) if (!after_ids.count(id)) destroyed.push_back(id);

        std::set<Edge> edges_after;
        std::set<Edge> removed;
        for (auto& [u, v] : changes.edge_removals) removed.insert(norm(u, v));
        for (auto& e : s.edges_before) {
            if (removed.count(e)) continue;
            if (std::find(destroyed.begin(), destroyed.end(), e.first) != destroyed.end()) continue;
            if (std::find(destroyed.begin(), destroyed.end(), e.second) != destroyed.end()) continue;
            edges_after.insert(e);
        }
        for (auto& [u, v] : changes.edge_updates) edges_after.insert(norm(u, v));

        auto list = [&](const std::vector<std::string>& xs) {
            os << "[";
            for (std::size_t i = 0; i < xs.size(); ++i) { if (i) os << ","; os << xs[i]; }
            os << "]";
        };
        auto ids = [&](const std::vector<std::size_t>& xs) {
            os << "[";
            for (std::size_t i = 0; i < xs.size(); ++i) { if (i) os << ","; os << xs[i]; }
            os << "]";
        };
        auto edges = [&](const std::set<Edge>& es) {
            os << "[";
            bool first = true;
            for (auto& [a, b] : es) { if (!first) os << ","; first = false; os << "[" << a << "," << b << "]"; }
            os << "]";
        };

        os << "{\"rule\":\"" << s.rule << "\",\"sim_step\":" << sim_step
           << ",\"ssa_step\":" << ssa_step << ",\"propensity\":";
        jnum(os, s.propensity);
        os << ",\"match\":{";
        for (std::size_t i = 0; i < s.match.size(); ++i) {
            if (i) os << ",";
            os << "\"" << s.match[i].first << "\":" << s.match[i].second;
        }
        os << "},\"before\":"; list(s.before);
        os << ",\"after\":";  list(after);
        os << ",\"context\":"; list(s.context);
        os << ",\"created\":"; ids(created);
        os << ",\"destroyed\":"; ids(destroyed);
        os << ",\"edges_before\":"; edges(s.edges_before);
        os << ",\"edges_after\":"; edges(edges_after);
        os << ",\"locals\":{";
        for (std::size_t i = 0; i < locals.size(); ++i) {
            if (i) os << ",";
            os << "\"" << locals[i].first << "\":" << locals[i].second;
        }
        os << "}}\n";

        auto& st = state();
        std::lock_guard<std::mutex> lk(st.mu);
        st.out << os.str();
        st.out.flush();
        ++st.count;
    }

}} // namespace DGGML::Trace

#endif // DGGML_RULE_TRACE_HPP
