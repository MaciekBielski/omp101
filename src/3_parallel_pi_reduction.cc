#include <chrono>
#include <cstdio>

#include <omp.h>

constexpr size_t steps = 1000000000;
constexpr double stepWidth = 1/static_cast<double>(steps);

namespace ch = std::chrono;

void parallelPiReduction() {
    auto start = ch::high_resolution_clock::now();
    double sum = 0.0;

    // parallel:    create a team of threads
    // for:         iterations of loops will be executed in parallel, first 'i' is private implicitly
    // reduction:   each thread will get private copy of the value, all copies are combined together at the end
    // schedule (optional): static is default, dynamic creates a runtime queue, auto means that compiler makes the choice
    #pragma omp parallel for schedule(auto) reduction(+:sum)
    for (int i = 0; i < steps; i++) {
        auto x = (double(i) + 0.5) * stepWidth;
        sum += 4.0/(1.0 + x*x);
    }
    // NOTE: implicit barrier at the end of the for loop, add `nowait` to disable it
    // NOTE: `nowait` impossible here as it is also the end of parallel{} region, which cannot be `nowait`-ed
    sum *= stepWidth;
    auto duration = ch::high_resolution_clock::now() - start;

    printf("ParallelFor: %.16lf\n", sum);
    printf("Duration: %lu ms\n", ch::duration_cast<ch::milliseconds>(duration).count());
}

int main(int argc, char const *argv[])
{
    parallelPiReduction();
    return 0;
}