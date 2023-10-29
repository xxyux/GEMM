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