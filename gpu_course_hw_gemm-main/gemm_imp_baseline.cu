#include "gemm.cuh"
#include "utils.cuh"
#include <cuda.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include "gemm_kernel.cuh"
#include <cuda_runtime.h>
#include <sys/time.h>
using std::vector;
using std::cout;


using namespace nvcuda;

std::string version;
FILE* file;

float time_diff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1e9 + 1e3 * (end->tv_usec - start->tv_usec);
}

class GEMM_Baseline : public GEMM<half, half, float>  {
public:
    GEMM_Baseline(int M, int N, int K) : GEMM(M, N, K) {}

    virtual void call_kernel() override {
        int T = 100;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        while(T--) {
            if(version == "baseline") {
                const int block_size = 16;
                dim3 threads(block_size, block_size);
                dim3 grid((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
                gemm_baseline<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
            }
            if(version == "tilegemm") {
                const int block_size = 16;
                dim3 threads(block_size, block_size);
                dim3 grid((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
                tilegemm<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
            }
            if(version == "v0") {
                simple_wmma<<< (((M*N)/(16*16))*32)/256, 256>>>(d_A, d_B, d_C, M, N, K);
            }
            if(version == "v1") {
                dim3 g(M/256,N/128);
                sharedmem_wmma_v1<<<g, 256>>>(d_A, d_B, d_C, M, N, K);
            }
            if(version == "v1_1") {
                dim3 g(M/256,N/128);
                sharedmem_wmma_v1_1<<<g, 256>>>(d_A, d_B, d_C, M, N, K);
            }
            if(version =="v3") {
                const int BM = 128, BN = 256;
                dim3 blockDim(256);
                int BX = (N + BN - 1) / BN;
                int BY = (M + BM - 1) / BM;
                dim3 gridDim(BX, BY);
                sharedmem_wmma_v3<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            }

            if(version == "v3_1") {
                const int BM = 128, BN = 256;
                dim3 blockDim(256);
                int BX = (N + BN - 1) / BN;
                int BY = (M + BM - 1) / BM;
                dim3 gridDim(BX, BY);
                sharedmem_wmma_v3_1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
            }

            if(version == "v3_2") {
                const int BM = 128, BN = 256, BK = 32;
                dim3 blockDim(256);
                int BX = (N + BN - 1) / BN;
                int BY = (M + BM - 1) / BM;
        
                dim3 gridDim(BX, BY);
        
                cudaFuncSetAttribute(sharedmem_wmma_v3_2,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
                unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
                sharedmem_wmma_v3_2<<<gridDim, blockDim, dsmem>>>(d_A, d_B, d_C, M, N, K);
            }
            cudaDeviceSynchronize();
        }
        gettimeofday(&end,NULL);
        fprintf(file, "%s, %d, %f\n", version.c_str(), N, time_diff(&start, &end)/100);
        // configure and call your kernel here. 
        // you may also add your timer in here
        // you can also add timers elsewhere. 
    }
};
 
int main(int argc, char *argv[]) {

    int Size = atoi(argv[1]);
    version = argv[2];
    GEMM_Baseline gemm(Size, Size, Size);
    file = fopen("output.csv", "a");

    bool correct = gemm.gemm();
    fclose(file);

    if (correct) {
        cout << "correct" << endl;
    } else {
        cout << "incorrect" << endl;
    }
    return 0;
}
