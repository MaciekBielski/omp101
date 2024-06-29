#include <chrono>
#include <cstdio>

#include <omp.h>

constexpr size_t steps = 1000000000;
constexpr double stepWidth = 1/static_cast<double>(steps);

namespace ch = std::chrono;

void parallelPiCritical() {
    double sum = 0.0;
    int threadsNb = omp_get_num_procs();
    omp_set_num_threads(threadsNb);

    auto start = ch::high_resolution_clock::now();
    #pragma omp parallel shared(sum, threadsNb)
    {
        auto tid = omp_get_thread_num();
        // #pragma omp master      // only tid==0 does this
        // {
        //     threadsNb = omp_get_num_threads();
        // }
        // #pragma omp barrier

        // Could be even simpler but:
        // - no guarantee which thread does the work
        // - worksharing (!) -> implicit barrier at the end 
        #pragma omp single
        {
            threadsNb = omp_get_num_threads();
        }

        const auto threadStride = stepWidth * threadsNb;
        double subSum = 0.0;

        for (auto x = (tid + 0.5) * stepWidth; x < 1.0; x += threadStride) {
            subSum += 4.0/(1.0 + x*x);
        }
        #pragma omp atomic
        sum += subSum * stepWidth;
    }
    auto duration = ch::high_resolution_clock::now() - start;

    printf("Threads: %d\n", threadsNb);
    printf("ParallelCritical: %.16lf\n", sum);
    printf("Duration: %lu ms\n", ch::duration_cast<ch::milliseconds>(duration).count());
}



int main(int argc, char const *argv[])
{
    parallelPiCritical();
    return 0;
}
