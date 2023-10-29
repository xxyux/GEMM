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