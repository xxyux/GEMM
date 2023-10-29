#ifndef HOMEWORK_UTILS_H
#define HOMEWORK_UTILS_H

#include <vector>
#include <cuda.h>
#include <bits/stdc++.h>

using std::vector;


static const int WARP_SIZE = 32;


// error macro
#define CHECK_ERROR(err)    checkCudaError(err, __FILE__, __LINE__)
#define GET_LAST_ERR()      getCudaLastErr(__FILE__, __LINE__)

inline void getCudaLastErr(const char *file, const int line) {
    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess) {
        std::cerr << "CUDA error: " << file << "(" << line << "): " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

inline void checkCudaError(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << file << "(" << line << "): " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

// convenient timers
#define TIMERSTART(label)                                                    \
        cudaSetDevice(0);                                                    \
        cudaEvent_t start##label, stop##label;                               \
        float time##label;                                                   \
        cudaEventCreate(&start##label);                                      \
        cudaEventCreate(&stop##label);                                       \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                     \
        cudaSetDevice(0);                                                    \
        cudaEventRecord(stop##label, 0);                                     \
        cudaEventSynchronize(stop##label);                                   \
        cudaEventElapsedTime(&time##label, start##label, stop##label);       \
        std::cout << "#" << time##label                                      \
                  << " ms (" << #label << ")" << std::endl;









#endif
