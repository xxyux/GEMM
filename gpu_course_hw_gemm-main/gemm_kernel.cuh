using namespace nvcuda;

// baseline. native version
template<typename Type_A, typename Type_B, typename Type_C>
__global__ void gemm_baseline(const  Type_A * __restrict__ A, const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    // you can change everything in this function, including the function signature
    // You can create a CUDA (.cu) file containing a class that inherits from the abstract base class GEMM.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if( col < M && row < N) {
        for(int i =0; i<K; i++ ) {
            sum += __half2float(A[row*N + i]) * __half2float(B[i*N + col]);
        }
        C[row*N + col] = sum;
    }
}
// TileGEMM
template<typename Type_A, typename Type_B, typename Type_C>
__global__ void tilegemm(const Type_A * __restrict__ A, const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;

    const int TILE_DIM = 16;
    Type_C sum=0;
    __shared__ Type_A a_tile[TILE_DIM][TILE_DIM];
    __shared__ Type_B b_tile[TILE_DIM][TILE_DIM];
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
        a_tile[ty][tx] = A[row * K + (tile*TILE_DIM) + tx];
        b_tile[ty][tx] = B[(tile*TILE_DIM+ty) * N + col];
        __syncthreads();
        
    #pragma unroll
        for(int i=0; i<TILE_DIM; i++) {
            sum += __half2float(a_tile[ty][i]) * __half2float(b_tile[i][tx]);
        }
        __syncthreads();
    }
    C[row*N + col] = sum;
}
// use tensor core to implement GEMM
template<typename Type_A, typename Type_B, typename Type_C>
__global__ void simple_wmma(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    // Every warp process 16*16
    int warp_in_a_row = K / 16;
    int warp_id_global = (blockIdx.x*blockDim.x + threadIdx.x) / 32;

    int warp_id_x = warp_id_global / warp_in_a_row;
    int warp_id_y = warp_id_global % warp_in_a_row;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int rowId = 16 * warp_id_x;
    int colId = 16 * warp_id_y;

    for(int kk=0; kk<K; kk+=16) {
        wmma::load_matrix_sync(a_frag, A + rowId * K + kk, K);
        wmma::load_matrix_sync(b_frag, B + kk * N + colId, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + rowId * N + colId, c_frag, N, wmma::mem_row_major);
}

// Shared memory [when threads_per_block is big, run error]
// 256*256 threads per block
template<typename Type_A, typename Type_B, typename Type_C>
__global__ void sharedmem_wmma(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    WHY is 64 and 128?
    BECAUSE the block's size is 1024*1024/256*32/(16*16) = 512 threads, the warp's/submatrix's number is 512 / 32 reshape(4 * 4).
    The block's size is (4*16)*(4*16) = 64*64.
    The row number is 64
    AND shared memory per block can incloud 64*64*6 half values
    The column number is 128 ( (64*128) *2 < 64*64*6 ) 
    */

    int block_lenth = N / gridDim.x; //64
    int block_width = M / gridDim.y; //64

    __shared__ Type_A A_shared[64*128];
    __shared__ Type_B B_shared[128*64];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int TILE_DIM = 128;

    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; // 16 (gridsize is 16 * 16)
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 16; // 4 (if block_width = 64)
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

// #pragma unroll
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
    // #pragma unroll
        for(int i=0; i < 16; i++) { // (64*128) / 512
            int id = tid + thread_in_block * i;
            int x_s = id / 128, y_s = id % 128;
            int x_A = block_firstrow_in_global + x_s;
            int y_A = y_s + tile * TILE_DIM;
            A_shared[x_s*128+y_s] = A[x_A * K + y_A];
            
            x_s = id / 64, y_s = id % 64;
            int x_B = x_s + tile * TILE_DIM;
            int y_B = block_firstcol_in_global + y_s;
            B_shared[x_s*64+y_s] = B[x_B * N + y_B];
        }
        __syncthreads();
    #pragma unroll
        for(int kk=0; kk < TILE_DIM; kk+=16) {
            wmma::load_matrix_sync(a_frag, A_shared + (warp_id_x*16) * 128 + kk , 128);
            wmma::load_matrix_sync(b_frag, B_shared + kk * 64 + (warp_id_y*16) , 64);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads(); 
    }
    int rowId = block_firstrow_in_global + warp_id_x*16;
    int colId = block_firstcol_in_global + warp_id_y*16;
    wmma::store_matrix_sync(C + rowId * N + colId, c_frag, N, wmma::mem_row_major);
}

// Shared memory version 1.0 
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

__global__ void sharedmem_wmma_v1(const half *__restrict__ A,
    const half *__restrict__ B, float *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    Shared memory per block can incloud 64*64*6(128*64 + 256*64) half values 
    */

    int block_lenth = 256;
    int block_width = 128;

    __shared__ half A_shared[256*64];
    __shared__ half B_shared[64*128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][4];
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    const int TILE_DIM = 64;

    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 64; 
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

// #pragma unroll
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
        for(int i=0; i<64; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 64, y_s = id % 64;
            int x_A = block_firstrow_in_global + x_s;
            int y_A = y_s + tile * TILE_DIM;
            A_shared[x_s*64+y_s] = A[x_A * K + y_A];
        }
        for(int i=0; i<32; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 128, y_s = id % 128;
            int x_B = x_s + tile * TILE_DIM;
            int y_B = block_firstcol_in_global + y_s;
            B_shared[x_s*128+y_s] = B[x_B * N + y_B];
        }
        __syncthreads();

        for(int i=0;i<4;i++) {
            for(int j=0;j<4;j++) {
                for(int kk=0;kk<TILE_DIM;kk+=16) {
                    int tx = warp_id_x*64+i*16;
                    int ty = warp_id_y*64+j*16;
                    wmma::load_matrix_sync(a_frag, A_shared + tx * 64 + kk , 64);
                    wmma::load_matrix_sync(b_frag, B_shared + kk * 128 + ty , 128);
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }
        __syncthreads(); 
    }
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            int rowId = block_firstrow_in_global + warp_id_x*64+i*16;
            int colId = block_firstcol_in_global + warp_id_y*64+j*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}


// version 1.1 (no shared memory)
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

__global__ void sharedmem_wmma_v1_1(const half *__restrict__ A,
    const half *__restrict__ B, float *__restrict__ C, size_t M, size_t N, size_t K) {

    int block_lenth = 256;
    int block_width = 128;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][4];
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    const int TILE_DIM = 64;

    int tid = threadIdx.x;
    // int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 64; 
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

// #pragma unroll
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
        for(int i=0;i<4;i++) {
            for(int j=0;j<4;j++) {
                for(int kk=0;kk<TILE_DIM;kk+=16) {
                    int tx = block_firstrow_in_global + warp_id_x*64 + i*16;
                    int ty = block_firstcol_in_global + warp_id_y*64 + j*16;
                    wmma::load_matrix_sync(a_frag, A + tx * K + (tile * TILE_DIM + kk), K);
                    wmma::load_matrix_sync(b_frag, B + (tile * TILE_DIM + kk) * N + ty, N);
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }
        __syncthreads(); 
    }
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            int rowId = block_firstrow_in_global + warp_id_x*64+i*16;
            int colId = block_firstcol_in_global + warp_id_y*64+j*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// Shared memory version 1.2 [first load data] and then [compute]
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

template<typename Type_A, typename Type_B, typename Type_C>
__global__ void sharedmem_wmma_v1_2(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    Shared memory per block can incloud 64*64*6(128*64 + 256*64) half values 
    */

    int block_lenth = 256;
    int block_width = 128;

    __shared__ Type_A A_shared[256*64];
    __shared__ Type_B B_shared[64*128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag[4][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag[4][4];
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    const int TILE_DIM = 64;

    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 64; 
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

// #pragma unroll
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
        for(int i=0; i<64; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 64, y_s = id % 64;
            int x_A = block_firstrow_in_global + x_s;
            int y_A = y_s + tile * TILE_DIM;
            A_shared[x_s*64+y_s] = A[x_A * K + y_A];
        }
        for(int i=0; i<32; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 128, y_s = id % 128;
            int x_B = x_s + tile * TILE_DIM;
            int y_B = block_firstcol_in_global + y_s;
            B_shared[x_s*128+y_s] = B[x_B * N + y_B];
        }
        __syncthreads();

        for(int kk=0;kk<TILE_DIM;kk+=16) {
            for(int i=0;i<4;i++) {
                for(int j=0;j<4;j++) {
                    int tx = warp_id_x*64+i*16;
                    int ty = warp_id_y*64+j*16;
                    wmma::load_matrix_sync(a_frag[i][j], A_shared + tx * 64 + kk , 64);
                    wmma::load_matrix_sync(b_frag[i][j], B_shared + kk * 128 + ty , 128);
                }
            }
            for(int i=0;i<4;i++) 
                for(int j=0;j<4;j++) 
                    wmma::mma_sync(c_frag[i][j], a_frag[i][j], b_frag[i][j], c_frag[i][j]);
        }
        __syncthreads(); 
    }
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            int rowId = block_firstrow_in_global + warp_id_x*64+i*16;
            int colId = block_firstcol_in_global + warp_id_y*64+j*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// Shared memory version 1.3 [add unroll]
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

template<typename Type_A, typename Type_B, typename Type_C>
__global__ void sharedmem_wmma_v1_3(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    Shared memory per block can incloud 64*64*6(128*64 + 256*64) half values 
    */

    int block_lenth = 256;
    int block_width = 128;

    __shared__ Type_A A_shared[256*64];
    __shared__ Type_B B_shared[64*128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag[4][4];
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    const int TILE_DIM = 64;

    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 64; 
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

// #pragma unroll
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
        #pragma unroll
        for(int i=0; i<64; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 64, y_s = id % 64;
            int x_A = block_firstrow_in_global + x_s;
            int y_A = y_s + tile * TILE_DIM;
            A_shared[x_s*64+y_s] = A[x_A * K + y_A];
        }
        #pragma unroll
        for(int i=0; i<32; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 128, y_s = id % 128;
            int x_B = x_s + tile * TILE_DIM;
            int y_B = block_firstcol_in_global + y_s;
            B_shared[x_s*128+y_s] = B[x_B * N + y_B];
        }
        __syncthreads();

        for(int i=0;i<4;i++) {
            for(int j=0;j<4;j++) {
                #pragma unroll
                for(int kk=0;kk<TILE_DIM;kk+=16) {
                    int tx = warp_id_x*64+i*16;
                    int ty = warp_id_y*64+j*16;
                    wmma::load_matrix_sync(a_frag, A_shared + tx * 64 + kk , 64);
                    wmma::load_matrix_sync(b_frag, B_shared + kk * 128 + ty , 128);
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }
        __syncthreads(); 
    }
    #pragma unroll
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            int rowId = block_firstrow_in_global + warp_id_x*64+i*16;
            int colId = block_firstcol_in_global + warp_id_y*64+j*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// Shared memory version 1.4 [bank conflicts, adjust shared_memory_size]
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

template<typename Type_A, typename Type_B, typename Type_C>
__global__ void sharedmem_wmma_v1_4(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    Shared memory per block can incloud 64*64*6(128*64 + 256*64) half values 
    */

    int block_lenth = 256;
    int block_width = 128;

    __shared__ Type_A A_shared[256*(32)];
    __shared__ Type_B B_shared[32*(128)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag;//256
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag;//256
    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag[4][4];//256*16 = 4096, sum = 256*18, A_shared + B_shared is 128*64+256*46 > 256*48+32*(128+16) 
#pragma unroll
    for(int i=0; i<4; i++) {
#pragma unroll
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    const int TILE_DIM = 32;

    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 64; 
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

#pragma unroll 32
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
// #pragma unroll
        for(int i=0; i<32; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 32, y_s = id % 32;
            int x_A = block_firstrow_in_global + x_s;
            int y_A = y_s + tile * TILE_DIM;
            
            A_shared[x_s*(32)+y_s] = A[x_A * K + y_A];
        }
// #pragma unroll
        for(int i=0; i<16; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 128, y_s = id % 128;
            int x_B = x_s + tile * TILE_DIM;
            int y_B = block_firstcol_in_global + y_s;
            
            B_shared[x_s*(128)+y_s] = B[x_B * N + y_B];
        }
        __syncthreads();

// #pragma unroll 
        for(int i=0;i<4;i++) {
// #pragma unroll
            for(int j=0;j<4;j++) {
                for(int kk=0;kk<TILE_DIM;kk+=16) {
                    int tx = warp_id_x*64+i*16;
                    int ty = warp_id_y*64+j*16;
                    wmma::load_matrix_sync(a_frag, A_shared + tx * (32) + kk , (32));
                    wmma::load_matrix_sync(b_frag, B_shared + kk * (128) + ty , (128));
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }
    }
// #pragma unroll
    for(int i=0;i<4;i++) {
// #pragma unroll
        for(int j=0;j<4;j++) {
            int rowId = block_firstrow_in_global + warp_id_x*64+i*16;
            int colId = block_firstcol_in_global + warp_id_y*64+j*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}


// Shared memory version 1.5 [unsolve bank conflicts, adjust shared_memory_size] vs 1.4
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

template<typename Type_A, typename Type_B, typename Type_C>
__global__ void sharedmem_wmma_v1_5(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    Shared memory per block can incloud 64*64*6(128*64 + 256*64) half values 
    */

    int block_lenth = 256;
    int block_width = 128;

    __shared__ Type_A A_shared[256*(32)];
    __shared__ Type_B B_shared[32*(128)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag;//256
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag;//256
    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag[4][4];//256*16 = 4096, sum = 256*18, A_shared + B_shared is 128*64+256*46 > 256*48+32*(128+16) 

    for(int i=0; i<4; i++) {
        #pragma unroll
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    const int TILE_DIM = 32;

    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_in_a_row = block_width / 64; 
    int warp_id_block = tid / 32;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

// #pragma unroll
    for(int tile = 0; tile < K/TILE_DIM; tile++) {
        for(int i=0; i<32; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 32, y_s = id % 32;
            int x_A = block_firstrow_in_global + x_s;
            int y_A = y_s + tile * TILE_DIM;
            
            A_shared[x_s*(32)+y_s] = A[x_A * K + y_A];
        }
        for(int i=0; i<16; i++) {
            int id = tid + thread_in_block * i;
            int x_s = id / 128, y_s = id % 128;
            int x_B = x_s + tile * TILE_DIM;
            int y_B = block_firstcol_in_global + y_s;
            
            B_shared[x_s*(128)+y_s] = B[x_B * N + y_B];
        }
        __syncthreads();

        for(int kk=0;kk<TILE_DIM;kk+=16) {
            for(int i=0;i<4;i++) {
                for(int j=0;j<4;j++) {
                    int tx = warp_id_x*64+i*16;
                    int ty = warp_id_y*64+j*16;
                    wmma::load_matrix_sync(a_frag, A_shared + tx * (32) + kk , (32));
                    wmma::load_matrix_sync(b_frag, B_shared + kk * (128) + ty , (128));
                }
            }
            
            for(int i=0;i<4;i++) {
                for(int j=0;j<4;j++) {
                        wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }
    }
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            int rowId = block_firstrow_in_global + warp_id_x*64+i*16;
            int colId = block_firstcol_in_global + warp_id_y*64+j*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

// Shared memory version 2.0 [adjust shared_memory_size, adjust tensor core]
// matrix size in block is [256,128]
// matrix size in warp is [64,64]

template<typename Type_A, typename Type_B, typename Type_C>
__global__ void sharedmem_wmma_v2(const Type_A *__restrict__ A,
    const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    
    /*
    Shared memory per block can incloud 64*64*6(128*64 + 256*64) half values 
    */

    int block_lenth = 256;
    int block_width = 128;

    __shared__ Type_A A_shared[64*(128+16)];
    __shared__ Type_B B_shared[128*(32+16)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, Type_A, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, Type_B, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Type_C> c_frag;

    const int TILE_DIM = 128;
    int tid = threadIdx.x;
    int thread_in_block = blockDim.x;
    int block_in_grid = blockIdx.x*gridDim.y + blockIdx.y;
    int block_in_a_row = gridDim.y; 
    int block_firstrow_in_global = (block_in_grid / block_in_a_row) * block_lenth;
    int block_firstcol_in_global = (block_in_grid % block_in_a_row) * block_width;
    int warp_id_block = tid / 32;
    int warp_in_a_row = 2;
    int warp_id_x = warp_id_block / warp_in_a_row;
    int warp_id_y = warp_id_block % warp_in_a_row;

    for(int i=0; i<block_lenth; i+=64) {
        for(int j=0; j<block_width; j+=32) {
            wmma::fill_fragment(c_frag, 0.0f);
            int subblock_firstrow_in_global = block_firstrow_in_global + i;
            int subblock_firstcol_in_global = block_firstcol_in_global + j;
            for(int k=0; k<K; k+=TILE_DIM) {
                for(int t=0; t<32; t++) {
                    int id = tid + thread_in_block * t;
                    int x_s = id / 128, y_s = id % 128;
                    int x_A = subblock_firstrow_in_global + x_s;
                    int y_A = y_s + k;
                    A_shared[x_s*(128+16)+y_s] = A[x_A * K + y_A];
                }
                for(int t=0; t<16; t++) {
                    int id = tid + thread_in_block * t;
                    int x_s = id / 32, y_s = id % 32;
                    int x_B = x_s + k;
                    int y_B = subblock_firstcol_in_global + y_s;
                    
                    B_shared[x_s*(32+16)+y_s] = B[x_B * N + y_B];
                }
                __syncthreads();
                for(int t=0; t<TILE_DIM; t+=16) {
                    int tx = warp_id_x*16;
                    int ty = warp_id_y*16;
                    wmma::load_matrix_sync(a_frag, A_shared + tx * (128+16) + t, (128+16));
                    wmma::load_matrix_sync(b_frag, B_shared + t * (32+16) + ty , (32+16));
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            } 
            int rowId = subblock_firstrow_in_global + warp_id_x*16;
            int colId = subblock_firstcol_in_global + warp_id_y*16;
            wmma::store_matrix_sync(C + rowId * N + colId, c_frag, N, wmma::mem_row_major);
        }
    }
}
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// Shared memory version 3.0 
// matrix size in block is [128, 256]
// matrix size in warp is [64, 64]
// bank conflict

__global__ void sharedmem_wmma_v3(
    half * __restrict__ a, half * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        FLOAT4(s_a[load_a_smem_m    ][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr        ]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr +     K]);
        FLOAT4(s_b[load_b_smem_k    ][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr        ]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr +     N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

// Shared memory verison 3.1
// cp.async.ca.shared.global

__global__ void sharedmem_wmma_v3_1(
    half * __restrict__ a, half * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a[0]);
    int s_b_base_addr = __cvta_generic_to_shared(s_b[0]);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

// Shared memory version 3.2 
// Double Buffer

__global__ void sharedmem_wmma_v3_2(
    half * __restrict__ a, half * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&b[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr        ]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr +     N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)), "l"(&b[load_b_gmem_addr + 3 * N]));

        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) +  0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64     ) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset +                    comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64     ], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}
