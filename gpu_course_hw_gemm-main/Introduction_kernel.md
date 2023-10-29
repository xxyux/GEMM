```
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
```
all threads_in_block is 256
1. baseline
   Every thread process 1 element in Matrix C
2. tilegemm
   use shared memory
   do tile, Tile_DIM = 32
3. v0
   simple_wmma
   no shared memory
   every warp do 16*16 elements in Matrix C
4. v1
   tensor core
   use shared memory
   do tile, Tile_DIM = 64
5. v1.1 
   vs v1
   tensor core
   no shared memory
   do tile, Tile_DIM = 64
6. v3
   vs v0
   tensor core
   shared memory
   do tile, Tile_DIM = 32
7. v3.1
   vs v3
   tensor core
   shared memory
   async.cp
8. v3.2
   vs v3.1
   tensor core 
   shared memory
   double buffer
