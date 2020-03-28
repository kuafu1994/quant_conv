//
// Created by PENGFEI ZHANG on 2020/3/26.
//

#include <stdint.h>
#include "params.h"
#include <iostream>

#include "qconv_kernel.h"

namespace quant_conv {

    // This is an experiments for 8-bit activation and 8-bit weight.
    void compute_quant_kernel(
            size_t mr,
            size_t nr,
            size_t kc, // input_channels
            size_t ks, // kernel_size
            const int8_t** a, // indirection_buffer
            const void* w, // packed weight
            int32_t* c, // output
            size_t c_stride, // output_channels
            struct qconv_neon_params qconv_params
    ){

        // The asm kernel below has the following NEON register allocation:
        // v16 - v31 are int32 accumulators.
        // so it can output 4 * 16 = 64 accumulators.
        // During accumulation, v0 -- v3 are used to load int8 data from the input
        // and v4 -- v7 from weight.
        // In this asm, we have to set the mr as 4 and nr as 16.
        //
        //
        //                      /-----------------------------
        //                      | v16.4s        ...   v28.4s | ch1
        //                      | v17.4s        ...   v29.4s | ch2
        //                      | v18.4s        ...   v30.4s | ch3
        //                      | v19.4s        ...   v31.4s | ch4
        //                      / ----------------------------
        //
        // Also, we assume that kc is a multiple of 8.

        int32_t* c0 = c;
        int32_t* c1 = c0 + c_stride;
        int32_t* c2 = c1 + c_stride;
        int32_t* c3 = c2 + c_stride;

        const int8_t* b_zero_point = (const int8_t*) &(qconv_params.kernel_zero_point);

        const int32_t* off = (const int32_t*) w;

        //std::cout << off[0] << ":" << off[1] << ":" << off[2] << ":" << off[3] << ":";


        // TODO, tackle zero_point.
        asm volatile (
                // clang-format off
                // make v16 -- v31 be zero.
                // They will be replaced when tackling kernel zero.

        // At first, w will added sizeof(int32_t) * nr
        // nr is 4
        // "ld1 {v16.4s}, [%[w]], #16\n"
        // Load the bytes from the input and weight
        "add %[p_w], %[p_w], #16\n"

        "eor v16.16b, v16.16b, v16.16b \n"
        "eor v17.16b, v17.16b, v17.16b \n"
        "eor v18.16b, v18.16b, v18.16b \n"
        "eor v19.16b, v19.16b, v19.16b \n"
        "eor v20.16b, v20.16b, v20.16b \n"
        "eor v21.16b, v21.16b, v21.16b \n"
        "eor v22.16b, v22.16b, v22.16b \n"
        "eor v23.16b, v23.16b, v23.16b \n"
        "eor v24.16b, v24.16b, v24.16b \n"
        "eor v25.16b, v25.16b, v25.16b \n"
        "eor v26.16b, v26.16b, v26.16b \n"
        "eor v27.16b, v27.16b, v27.16b \n"
        "eor v28.16b, v28.16b, v28.16b \n"
        "eor v29.16b, v29.16b, v29.16b \n"
        "eor v30.16b, v30.16b, v30.16b \n"
        "eor v31.16b, v31.16b, v31.16b \n"


        // Let x5 store the kernel_size
        "mov x5, %x[p_ks]              \n"

        // kernel_size loop.
        "0:                         \n"

        // Load the pointer for a0, a1, a2, a3.
        // Here, we must make mr as 4.
        "ldr x6, [%[p_a]], #8             \n"
        "ldr x7, [%[p_a]], #8             \n"
        "ldr x8, [%[p_a]], #8             \n"
        "ldr x9, [%[p_a]], #8             \n"

        // load 4 16-channel pixel from the input.
        "ld1 {v0.16b}, [x6], #16\n"
        "ld1 {v1.16b}, [x7], #16\n"
        "ld1 {v2.16b}, [x8], #16\n"
        "ld1 {v3.16b}, [x9], #16\n"

        // load 4 oc x 16 ic wieghts from w.
        // nr must be 4 and kr must be 16.
        "ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%[p_w]], #64\n"

        // w4 = kc / 16
        "lsr w4, %w[p_kc], #4 \n"

        // perform the first few multiply-adds on the data
        // that we already loaded
        // v8, v9, v10 and v11 are for different pixels but same oc
        // v8, v12 are for same pixel but different oc.
        "smull v8.8h, v0.8b, v4.8b\n"
        "smull v9.8h, v1.8b, v4.8b\n"
        "smull v10.8h, v2.8b, v4.8b\n"
        "smull v11.8h, v3.8b, v4.8b\n"
        "smull v12.8h, v0.8b, v5.8b\n"
        "smull v13.8h, v1.8b, v5.8b\n"
        "smull v14.8h, v2.8b, v5.8b\n"
        "smull v15.8h, v3.8b, v5.8b\n"

        "smlal2 v8.8h, v0.16b, v4.16b \n"
        "smlal2 v9.8h, v1.16b, v4.16b \n"
        "smlal2 v10.8h, v2.16b, v4.16b \n"
        "smlal2 v11.8h, v3.16b, v4.16b \n"
        "smlal2 v12.8h, v0.16b, v5.16b \n"
        "smlal2 v13.8h, v1.16b, v5.16b \n"
        "smlal2 v14.8h, v2.16b, v5.16b \n"
        "smlal2 v15.8h, v3.16b, v5.16b \n"

        "subs w4, w4, #1\n"
        "beq 2f\n"

        "1:             \n"

        "sadalp v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[p_w]], #16 \n"
        "smull v8.8h, v0.8b, v6.8b  \n"

        "sadalp v17.4s, v9.8h       \n"
        "ld1 {v5.16b}, [%[p_w]], #16 \n"
        "smull v9.8h, v1.8b, v6.8b \n"

        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"

        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"

        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"

        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"

        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"

        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "ld1 {v6.16b}, [%[p_w]], #16      \n"

        "smlal2 v12.8h, v0.16b, v7.16b \n"
        "ld1 {v0.16b}, [x6], #16   \n"

        "smlal2 v13.8h, v1.16b, v7.16b \n"
        "ld1 {v1.16b}, [x7], #16   \n"

        "smlal2 v14.8h, v2.16b, v7.16b\n"
        "ld1 {v2.16b}, [x8], #16   \n"

        "smlal2 v15.8h, v3.16b, v7.16b \n"
        "ld1 {v3.16b}, [x9], #16      \n"

        "sadalp v24.4s, v8.8h         \n"
        "smull v8.8h, v0.8b, v4.8b \n"

        "sadalp v25.4s, v9.8h \n"
        "smull v9.8h, v1.8b, v4.8b \n"

        "sadalp v26.4s, v10.8h \n"
        "smull v10.8h, v2.8b, v4.8b\n"

        "sadalp v27.4s, v11.8h\n"
        "smull v11.8h, v3.8b, v4.8b\n"

        "ld1 {v7.16b}, [%[p_w]], #16\n"

        "sadalp v28.4s, v12.8h\n"
        "smull v12.8h, v0.8b, v5.8b \n"

        "sadalp v29.4s, v13.8h\n"
        "smull v13.8h, v1.8b, v5.8b \n"

        "sadalp v30.4s, v14.8h\n"
        "smull v14.8h, v2.8b, v5.8b \n"

        "sadalp v31.4s, v15.8h\n"
        "smull v15.8h, v3.8b, v5.8b \n"

        "subs w4, w4, #1\n"

        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        "bne 1b\n"

        "2:\n"
        "sadalp  v16.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "sadalp  v25.4s, v9.8h\n"
        "sadalp  v26.4s, v10.8h\n"
        "sadalp  v27.4s, v11.8h\n"
        "sadalp  v28.4s, v12.8h\n"
        "sadalp  v29.4s, v13.8h\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        "subs x5, x5, #1\n"
        "bne 0b\n"
        // End of accumulation.

        "ld1 {v0.4s}, [%[p_off]]\n"
        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v20.4s\n"
        "addp v17.4s, v17.4s, v21.4s\n"
        "addp v18.4s, v18.4s, v22.4s\n"
        "addp v19.4s, v19.4s, v23.4s\n"

        "addp v24.4s, v24.4s, v28.4s\n"
        "addp v25.4s, v25.4s, v29.4s\n"
        "addp v26.4s, v26.4s, v30.4s\n"
        "addp v27.4s, v27.4s, v31.4s\n"

        "addp v16.4s, v16.4s, v24.4s\n"
        "addp v17.4s, v17.4s, v25.4s\n"
        "addp v18.4s, v18.4s, v26.4s\n"
        "addp v19.4s, v19.4s, v27.4s\n"

        "add v16.4s, v16.4s, v0.4s\n"
        "add v17.4s, v17.4s, v0.4s\n"
        "add v18.4s, v18.4s, v0.4s\n"
        "add v19.4s, v19.4s, v0.4s\n"

        "st1 {v16.4s}, [%[p_c0]], #16\n"
        "st1 {v17.4s}, [%[p_c1]], #16\n"
        "st1 {v18.4s}, [%[p_c2]], #16\n"
        "st1 {v19.4s}, [%[p_c3]], #16\n"

        :
        [p_a] "+r" (a),
        [p_w] "+r" (w),
        [p_c0] "+r" (c0),
        [p_c1] "+r" (c1),
        [p_c2] "+r" (c2),
        [p_c3] "+r" (c3)
        :
        [p_kc] "r" (kc),
        [p_ks] "r" (ks),
        [p_off] "r" (off)
        :
          "cc", "memory", "x4", "x5", "x6", "x7", "x8", "x9",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    //    std::cout << c[0] << std::endl;
    }

}
