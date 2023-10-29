__shared__ half A_shared[256*64];
__shared__ half B_shared[64*128];
// -----initial fragment code------
for(int tile = 0; tile < K/TILE_DIM; tile++) { // TILE_DIM = 64
    for(int i=0; i<64; i++) {//process A_shared
        int id = tid + thread_in_block * i;
        int x_s = id / 64, y_s = id % 64;
        int x_A = block_firstrow_in_global + x_s;
        int y_A = y_s + tile * TILE_DIM;
        A_shared[x_s*64+y_s] = A[x_A * K + y_A];
    }
    for(int i=0; i<32; i++) {
        // process B_shared
    }
    __syncthreads();// synchronize, make sure all threads go correctly
    // ----compute code----
}