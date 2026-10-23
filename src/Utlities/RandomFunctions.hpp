#ifndef DGGML_RANDOMFUNCTIONS_HPP
#define DGGML_RANDOMFUNCTIONS_HPP

#include <random>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace DGGML {

// --- Global RNG seeding control (primarily for reproducible testing) ---
// Default behaviour is unchanged: each thread's engine is seeded from
// std::random_device (non-deterministic). Call set_global_seed()/seed_omp_threads()
// to switch to deterministic seeding before a run.
inline bool& rng_use_fixed_seed() { static bool v = false; return v; }
inline std::uint32_t& rng_base_seed() { static std::uint32_t s = 0u; return s; }

// Persistent per-thread Mersenne Twister. Unlike the previous implementation
// (which created and seeded a fresh engine on every single draw), this keeps one
// continuous stream per thread, which is both faster and statistically sound.
inline std::mt19937& thread_rng() {
    static thread_local std::mt19937 engine{
        rng_use_fixed_seed() ? std::mt19937(rng_base_seed())
                             : std::mt19937(std::random_device{}())};
    return engine;
}

// Reseed the calling thread's engine explicitly.
inline void seed_this_thread(std::uint32_t seed) { thread_rng().seed(seed); }

// Reseed the calling thread's engine from a deterministic mix of identifiers
// (base seed + step + dimension + geocell key). std::seed_seq decorrelates the
// resulting state so distinct geocells/steps get effectively independent streams.
// Used by the parallel runner to key randomness to *which* geocell is being
// processed rather than *which thread* runs it, making results reproducible and
// independent of OpenMP scheduling and thread count.
inline void seed_geocell_stream(std::uint32_t base, std::uint32_t step,
                                std::uint32_t dim, std::uint64_t cell) {
    std::seed_seq seq{ base, step, dim,
                       static_cast<std::uint32_t>(cell & 0xffffffffull),
                       static_cast<std::uint32_t>(cell >> 32) };
    thread_rng().seed(seq);
}

// Enable deterministic mode with a base seed and reseed the calling thread.
inline void set_global_seed(std::uint32_t seed) {
    rng_use_fixed_seed() = true;
    rng_base_seed() = seed;
    seed_this_thread(seed);
}

// Enable deterministic mode and reseed every thread in the current OpenMP team
// as (base + thread_id). Call this on the main thread before a run; OpenMP
// reuses the same worker threads across parallel regions, so their thread-local
// engines persist into the simulation's parallel regions.
inline void seed_omp_threads(std::uint32_t base) {
    rng_use_fixed_seed() = true;
    rng_base_seed() = base;
#ifdef _OPENMP
    #pragma omp parallel
    {
        seed_this_thread(base + static_cast<std::uint32_t>(omp_get_thread_num()));
    }
#else
    seed_this_thread(base);
#endif
}

// Return to non-deterministic (random_device) seeding for the calling thread.
inline void clear_global_seed() {
    rng_use_fixed_seed() = false;
    thread_rng().seed(std::random_device{}());
}

} // namespace DGGML

auto RandomRealsBetween(double low, double high)
{
    auto randomFunc =
            [distribution_ = std::uniform_real_distribution<double>(low, high)]() mutable
            {
                return distribution_(DGGML::thread_rng());
            };
    return randomFunc;
};

auto RandomIntsBetween(int low, int high)
{
    auto randomFunc =
            [distribution_ = std::uniform_int_distribution<int>(low, high)]() mutable
            {
                return distribution_(DGGML::thread_rng());
            };
    return randomFunc;
};

#endif //DGGML_RANDOMFUNCTIONS_HPP
