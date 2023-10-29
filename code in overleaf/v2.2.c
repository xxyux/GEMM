// in kernel loop, TILE_DIM = 32, bk is the number of tile 
for (int bk = 1; bk < K / BK; bk++) {
int smem_sel_next = ((bk - 1) & 1) ^ 1; // dimension of scrolling array
asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr        ]));
asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)), "l"(&a[load_a_gmem_addr +     K]));
// ---- load data to shared memory array_b ----
// ---- compute code ----
asm ("cp.async.commit_group;\n" ::);
asm ("cp.async.wait_group 0;\n" ::);}
