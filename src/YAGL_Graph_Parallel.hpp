#ifndef YAGL_GRAPH_PARALLEL_HPP
#define YAGL_GRAPH_PARALLEL_HPP
#pragma once

// YAGL_Graph_Parallel.hpp
//
// Provides YAGL::ParallelGraph<K,D>: a thin composition wrapper around
// YAGL::Graph<K,D> that adds a POSIX reader-writer lock (pthread_rwlock_t).
//
// Concurrency model:
//   - Multiple threads may hold ReadGuards simultaneously (concurrent DFS,
//     induced_subgraph, subgraph_isomorphism from different geocells).
//   - Only one thread may hold a WriteGuard (Phase-5 system_graph commit),
//     and it is exclusive with all readers.
//
// The lock is NOT held internally by ParallelGraph methods.  Callers are
// responsible for acquiring the appropriate guard for the lifetime of any
// iterator or reference obtained from the graph.
//
// For plain YAGL::Graph<K,D>, make_graph_read_guard / make_graph_write_guard
// return a NullLockGuard (zero overhead) so serial code compiles unchanged.

#include "YAGL_Graph.hpp"
#include <pthread.h>

namespace YAGL {

// ---------------------------------------------------------------------------
// Zero-overhead no-op guard returned for plain YAGL::Graph<K,D>.
// ---------------------------------------------------------------------------
struct NullLockGuard {};

// ---------------------------------------------------------------------------
// RAII write guard: acquires exclusive lock in ctor, releases in dtor.
// ---------------------------------------------------------------------------
struct WriteGuard {
    pthread_rwlock_t* lk;
    explicit WriteGuard(pthread_rwlock_t* l) : lk(l) { pthread_rwlock_wrlock(lk); }
    ~WriteGuard() { if (lk) pthread_rwlock_unlock(lk); }
    WriteGuard(const WriteGuard&) = delete;
    WriteGuard& operator=(const WriteGuard&) = delete;
    WriteGuard(WriteGuard&& o) noexcept : lk(o.lk) { o.lk = nullptr; }
    WriteGuard& operator=(WriteGuard&&) = delete;
};

// ---------------------------------------------------------------------------
// RAII read guard: acquires shared lock in ctor, releases in dtor.
// ---------------------------------------------------------------------------
struct ReadGuard {
    pthread_rwlock_t* lk;
    explicit ReadGuard(pthread_rwlock_t* l) : lk(l) { pthread_rwlock_rdlock(lk); }
    ~ReadGuard() { if (lk) pthread_rwlock_unlock(lk); }
    ReadGuard(const ReadGuard&) = delete;
    ReadGuard& operator=(const ReadGuard&) = delete;
    ReadGuard(ReadGuard&& o) noexcept : lk(o.lk) { o.lk = nullptr; }
    ReadGuard& operator=(ReadGuard&&) = delete;
};

// ---------------------------------------------------------------------------
// Free-function overloads for plain YAGL::Graph<K,D>: no-op guards.
// ADL finds these because Graph<K,D> lives in namespace YAGL.
// ---------------------------------------------------------------------------
template<typename K, typename D>
inline NullLockGuard make_graph_write_guard(Graph<K,D>&) { return {}; }

template<typename K, typename D>
inline NullLockGuard make_graph_read_guard(Graph<K,D>&) { return {}; }

// ---------------------------------------------------------------------------
// ParallelGraph<K,D>
//
// Exposes exactly the same public interface as YAGL::Graph<K,D> and delegates
// every method to the inner graph_.  Callers must hold the appropriate RAII
// guard across any sequence of calls that must be atomic.
// ---------------------------------------------------------------------------
template <typename KeyType, typename DataType>
class ParallelGraph {
public:
    // Re-export all typedefs so templated algorithms (induced_subgraph,
    // recursive_dfs, subgraph_isomorphism2) see the same names as Graph<K,D>.
    using key_type                    = typename Graph<KeyType,DataType>::key_type;
    using data_type                   = typename Graph<KeyType,DataType>::data_type;
    using node_type                   = typename Graph<KeyType,DataType>::node_type;
    using edge_type                   = typename Graph<KeyType,DataType>::edge_type;
    using node_set_type               = typename Graph<KeyType,DataType>::node_set_type;
    using edge_set_type               = typename Graph<KeyType,DataType>::edge_set_type;
    using in_out_nbr_type             = typename Graph<KeyType,DataType>::in_out_nbr_type;
    using adjacency_list_type         = typename Graph<KeyType,DataType>::adjacency_list_type;
    using node_list_type              = typename Graph<KeyType,DataType>::node_list_type;
    using edge_list_type              = typename Graph<KeyType,DataType>::edge_list_type;
    using counting_type               = typename Graph<KeyType,DataType>::counting_type;
    using node_iterator               = typename Graph<KeyType,DataType>::node_iterator;
    using const_node_iterator         = typename Graph<KeyType,DataType>::const_node_iterator;
    using edge_iterator               = typename Graph<KeyType,DataType>::edge_iterator;
    using const_edge_iterator         = typename Graph<KeyType,DataType>::const_edge_iterator;
    using adjacency_iterator          = typename Graph<KeyType,DataType>::adjacency_iterator;
    using const_adjacency_iterator    = typename Graph<KeyType,DataType>::const_adjacency_iterator;
    using node_set_nbr_iterator       = typename Graph<KeyType,DataType>::node_set_nbr_iterator;
    using const_node_set_nbr_iterator = typename Graph<KeyType,DataType>::const_node_set_nbr_iterator;

    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------
    ParallelGraph()  { pthread_rwlock_init(&rwlock_, nullptr); }
    ~ParallelGraph() { pthread_rwlock_destroy(&rwlock_); }

    // Copy: deep-copy the graph data; the copy gets its own fresh rwlock
    // (pthread_rwlock_t is not copyable and copying a lock state makes no sense).
    ParallelGraph(const ParallelGraph& o) : graph_(o.graph_) {
        pthread_rwlock_init(&rwlock_, nullptr);
    }
    ParallelGraph& operator=(const ParallelGraph& o) {
        if (this != &o) graph_ = o.graph_;
        return *this;
    }

    // Move: move the graph data; give the object a fresh rwlock.
    ParallelGraph(ParallelGraph&& o) noexcept : graph_(std::move(o.graph_)) {
        pthread_rwlock_init(&rwlock_, nullptr);
    }
    ParallelGraph& operator=(ParallelGraph&& o) noexcept {
        if (this != &o) graph_ = std::move(o.graph_);
        return *this;
    }

    // -----------------------------------------------------------------------
    // Guard factory: called by free functions below.
    // -----------------------------------------------------------------------
    WriteGuard write_guard() { return WriteGuard(&rwlock_); }
    ReadGuard  read_guard()  { return ReadGuard(&rwlock_);  }

    // -----------------------------------------------------------------------
    // Iterator access
    // -----------------------------------------------------------------------
    node_iterator      node_list_begin()  { return graph_.node_list_begin(); }
    node_iterator      node_list_end()    { return graph_.node_list_end();   }
    edge_iterator      edge_list_begin()  { return graph_.edge_list_begin(); }
    edge_iterator      edge_list_end()    { return graph_.edge_list_end();   }
    adjacency_iterator adjacency_list_begin() { return graph_.adjacency_list_begin(); }
    adjacency_iterator adjacency_list_end()   { return graph_.adjacency_list_end();   }

    node_set_nbr_iterator out_neighbors_begin(const node_type& n) { return graph_.out_neighbors_begin(n);   }
    node_set_nbr_iterator out_neighbors_end  (const node_type& n) { return graph_.out_neighbors_end(n);     }
    node_set_nbr_iterator out_neighbors_begin(KeyType key)        { return graph_.out_neighbors_begin(key); }
    node_set_nbr_iterator out_neighbors_end  (KeyType key)        { return graph_.out_neighbors_end(key);   }
    node_set_nbr_iterator in_neighbors_begin (const node_type& n) { return graph_.in_neighbors_begin(n);    }
    node_set_nbr_iterator in_neighbors_end   (const node_type& n) { return graph_.in_neighbors_end(n);      }
    node_set_nbr_iterator in_neighbors_begin (KeyType key)        { return graph_.in_neighbors_begin(key);  }
    node_set_nbr_iterator in_neighbors_end   (KeyType key)        { return graph_.in_neighbors_end(key);    }

    // -----------------------------------------------------------------------
    // Neighbor set access (returns reference into internal storage — caller
    // must hold a read guard for the lifetime of the reference)
    // -----------------------------------------------------------------------
    node_set_type& out_neighbors(const node_type& n) { return graph_.out_neighbors(n);   }
    node_set_type& in_neighbors (const node_type& n) { return graph_.in_neighbors(n);    }
    node_set_type& out_neighbors(KeyType key)        { return graph_.out_neighbors(key); }
    node_set_type& in_neighbors (KeyType key)        { return graph_.in_neighbors(key);  }

    // -----------------------------------------------------------------------
    // Edge operations
    // -----------------------------------------------------------------------
    bool adjacent(const KeyType& a, const KeyType& b) { return graph_.adjacent(a, b); }
    bool adjacent(const node_type& a, node_type& b)   { return graph_.adjacent(a, b); }

    void addEdge   (const node_type& a, const node_type& b) { graph_.addEdge(a, b);    }
    void addEdge   (KeyType a, KeyType b)                    { graph_.addEdge(a, b);    }
    void removeEdge(const node_type& a, const node_type& b) { graph_.removeEdge(a, b); }
    void removeEdge(KeyType a, KeyType b)                    { graph_.removeEdge(a, b); }

    // -----------------------------------------------------------------------
    // Node operations
    // -----------------------------------------------------------------------
    node_list_type  getNodeSet()    { return graph_.getNodeSet();    }
    node_list_type& getNodeSetRef() { return graph_.getNodeSetRef(); }

    void          addNode   (const node_type& n) { graph_.addNode(n);    }
    void          removeNode(const node_type& n) { graph_.removeNode(n); }
    node_iterator findNode  (KeyType k)          { return graph_.findNode(k); }

    // -----------------------------------------------------------------------
    // Graph metrics
    // -----------------------------------------------------------------------
    counting_type numNodes()                     { return graph_.numNodes();     }
    counting_type numEdges()                     { return graph_.numEdges();     }
    counting_type in_degree (const node_type& n) { return graph_.in_degree(n);  }
    counting_type out_degree(const node_type& n) { return graph_.out_degree(n); }
    counting_type degree    (const node_type& n) { return graph_.degree(n);     }
    counting_type min_degree()                   { return graph_.min_degree();   }
    counting_type max_degree()                   { return graph_.max_degree();   }
    double        avg_degree()                   { return graph_.avg_degree();   }

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------
    void reserve(std::size_t n) { graph_.reserve(n); }
    void clear()                { graph_.clear();     }

    bool isDirected()   { return graph_.isDirected();  }
    bool isUndirected() { return graph_.isUndirected(); }

    DataType& operator[](KeyType k) { return graph_[k]; }

    // Access the inner plain graph (e.g. for code that explicitly needs Graph<K,D>).
    Graph<KeyType, DataType>&       inner()       { return graph_; }
    const Graph<KeyType, DataType>& inner() const { return graph_; }

private:
    Graph<KeyType, DataType> graph_;
    mutable pthread_rwlock_t rwlock_;
};

// ---------------------------------------------------------------------------
// Free-function overloads for ParallelGraph<K,D>: return real locking guards.
// ADL finds these because ParallelGraph<K,D> lives in namespace YAGL.
// ---------------------------------------------------------------------------
template<typename K, typename D>
inline WriteGuard make_graph_write_guard(ParallelGraph<K,D>& g) { return g.write_guard(); }

template<typename K, typename D>
inline ReadGuard make_graph_read_guard(ParallelGraph<K,D>& g) { return g.read_guard(); }

// ---------------------------------------------------------------------------
// operator<< for ParallelGraph: mirrors the output of Graph<K,D>'s operator<<.
// ---------------------------------------------------------------------------
template<typename K, typename D>
inline std::ostream& operator<<(std::ostream& os, ParallelGraph<K,D>& g)
{
    os << "Current Graph Info: {\n"
       << "\tUndirected: " << g.isUndirected() << "\n"
       << "\t# of Nodes: " << g.numNodes() << "\n"
       << "\t# of Edges: " << g.numEdges() << "\n"
       << "\tMin Degree: " << g.min_degree() << "\n"
       << "\tMax Degree: " << g.max_degree() << "\n"
       << "\tAvg Degree: " << g.avg_degree() << "\n"
       << "}\n";
    return os;
}

} // namespace YAGL

#endif // YAGL_GRAPH_PARALLEL_HPP
