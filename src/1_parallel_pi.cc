#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <numeric>

#include <omp.h>

constexpr int threadsNb = 8;
constexpr size_t steps = 1000000000;
constexpr size_t subSteps = steps / threadsNb;
constexpr double stepWidth = 1/static_cast<double>(steps);
constexpr double subStepsWidth = stepWidth * subSteps;

namespace ch = std::chrono;

void sequentialPi() {
    double sum = 0.0;
    auto x = stepWidth * 0.5;

    auto start = ch::high_resolution_clock::now();
    for (size_t i = 0; i < steps; i++) {
        sum += 4.0/(1.0 + x*x);
        x += stepWidth;
    }
    sum *= stepWidth;
    auto duration = ch::high_resolution_clock::now() - start;

    printf("Sequential: %.16lf\n", sum);
    printf("Duration: %lu ms\n", ch::duration_cast<ch::milliseconds>(duration).count());
}

void parallelPi() {
    // We can get less threads than requested. This is not taken into account here.
    // This should be checked inside a parallel region by one of threads and saved to shared variable
    omp_set_num_threads(threadsNb);
    // This is a bit naive. It is possible that different threads will write to the same cache line,
    // thus invalidating each other's data. This is called false sharing.
    auto subSums = std::array<double, threadsNb>{};
    auto start = ch::high_resolution_clock::now();

    // Execute on a nb of threads
    #pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        double subSum = 0.0;                // do not use the std::array directly here, it kills the performance
        auto x = (subStepsWidth * tid) + stepWidth * 0.5;
        for (size_t i = 0; i < subSteps; i++ ) {
            subSum += 4.0/(1.0 + x*x);
            x += stepWidth;
        }
        subSums[tid] = subSum;              // using just here is apparently fine
    }

    double sum = std::accumulate(subSums.cbegin(), subSums.cend(), 0.0);
    sum *= stepWidth;
    auto duration = ch::high_resolution_clock::now() - start;
    
    printf("Parallel: %.16lf\n", sum);
    printf("Duration: %lu ms\n", ch::duration_cast<ch::milliseconds>(duration).count());
}

int main(int argc, char const *argv[])
{
    sequentialPi();
    parallelPi();
    return 0;
}
