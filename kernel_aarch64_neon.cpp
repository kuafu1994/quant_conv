//
// Created by PENGFEI ZHANG on 2020/3/26.
//

#include <stdint.h>
#include "params.h"
#include <iostream>
#include <string.h>

#include "qconv_kernel.h"

namespace quant_conv {
#if 0
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

        // const int32_t* off = (const int32_t*) w;

        int32_t *off = new int32_t[4];
        memcpy(off, w, 16);
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

#endif

#if 0
    void compute_quant_kernel_a7w7(
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

        //const int32_t* off = (const int32_t*) w;
        int32_t* off = new int32_t[4];
        memcpy(off, w, 16);

       // std::cout << off[0] << ":" << off[1] << ":" << off[2] << ":" << off[3] << std::endl;


        // TODO, tackle zero_point.
        asm volatile (

#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"
        // clang-format off
        // make v16 -- v31 be zero.
        // They will be replaced when tackling kernel zero.

        // At first, w will added sizeof(int32_t) * nr
        // nr is 4
        // "ld1 {v16.4s}, [%[w]], #16\n"
        // Load the bytes from the input and weight
        "add %[p_w], %[p_w], #16\n"

        MAKE_ZERO(v16)
        MAKE_ZERO(v17)
        MAKE_ZERO(v18)
        MAKE_ZERO(v19)
        MAKE_ZERO(v20)
        MAKE_ZERO(v21)
        MAKE_ZERO(v22)
        MAKE_ZERO(v23)
        MAKE_ZERO(v24)
        MAKE_ZERO(v25)
        MAKE_ZERO(v26)
        MAKE_ZERO(v27)
        MAKE_ZERO(v28)
        MAKE_ZERO(v29)
        MAKE_ZERO(v30)
        MAKE_ZERO(v31)

#undef MAKE_ZERO

        "ld1r {v6.16b}, [%[zero_point]]\n"

        // Let x5 store the kernel_size
        "mov w5, %w[p_ks]              \n"

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

        // load 2 oc x 16 ic weights and substract the weight zero point.
        "ld1 {v4.16b, v5.16b}, [%[p_w]], #32\n"
        "sub v4.16b, v4.16b, v6.16b\n"
        "sub v5.16b, v5.16b, v6.16b\n"

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

        "ld1 {v4.16b, v5.16b}, [%[p_w]], #32\n" // This v4 is for v24, v25, v26, v27; v5 is for v28, v29, v30, v31
        "sub v4.16b, v4.16b, v6.16b\n"
        "sub v5.16b, v5.16b, v6.16b\n"

        "sadalp v16.4s, v8.8h\n"
        "sadalp v17.4s, v9.8h\n"
        "sadalp v18.4s, v10.8h\n"
        "sadalp v19.4s, v11.8h\n"

        "smull v8.8h, v0.8b, v4.8b  \n"
        "smull v9.8h, v1.8b, v4.8b \n"
        "smull v10.8h, v2.8b, v4.8b\n"
        "smull v11.8h, v3.8b, v4.8b\n"

        "sadalp  v20.4s, v12.8h\n"
        "smull   v12.8h, v0.8b, v5.8b\n"

        "sadalp  v21.4s, v13.8h\n"
        "smull   v13.8h, v1.8b, v5.8b\n"

        "sadalp  v22.4s, v14.8h\n"
        "smull   v14.8h, v2.8b, v5.8b\n"

        "sadalp  v23.4s, v15.8h\n"
        "smull   v15.8h, v3.8b, v5.8b\n"

        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "ld1 {v4.16b}, [%[p_w]], #16      \n" // This v4 is for next v16, v17, v18, v19.
        "sub v4.16b, v4.16b, v6.16b       \n"

        "smlal2 v12.8h, v0.16b, v5.16b \n"
        "ld1 {v0.16b}, [x6], #16   \n"

        "smlal2 v13.8h, v1.16b, v5.16b \n"
        "ld1 {v1.16b}, [x7], #16   \n"

        "smlal2 v14.8h, v2.16b, v5.16b\n"
        "ld1 {v2.16b}, [x8], #16   \n"

        "smlal2 v15.8h, v3.16b, v5.16b \n"
        "ld1 {v3.16b}, [x9], #16      \n"

        "sadalp v24.4s, v8.8h         \n"
        "smull v8.8h, v0.8b, v4.8b \n"

        "sadalp v25.4s, v9.8h \n"
        "smull v9.8h, v1.8b, v4.8b \n"

        "sadalp v26.4s, v10.8h \n"
        "smull v10.8h, v2.8b, v4.8b\n"

        "sadalp v27.4s, v11.8h\n"
        "smull v11.8h, v3.8b, v4.8b\n"

        "ld1 {v5.16b}, [%[p_w]], #16\n" // This v5 is for next v20, v21, v22, v23.
        "sub v5.16b, v5.16b, v6.16b \n"

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

        "ld1 {v4.16b, v5.16b}, [%[p_w]], #32\n" // This v4 is for v24, v25, v26, v27; v5 is for v28, v29, v30, v31
        "sub v4.16b, v4.16b, v6.16b\n"
        "sub v5.16b, v5.16b, v6.16b\n"

        "sadalp  v16.4s, v8.8h\n"
        "sadalp  v17.4s, v9.8h\n"
        "sadalp  v18.4s, v10.8h\n"
        "sadalp  v19.4s, v11.8h\n"

        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"

        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "sadalp  v25.4s, v9.8h\n"
        "sadalp  v26.4s, v10.8h\n"
        "sadalp  v27.4s, v11.8h\n"
        "sadalp  v28.4s, v12.8h\n"
        "sadalp  v29.4s, v13.8h\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        "subs w5, w5, #1\n"
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
        [p_off] "r" (off),
        [zero_point] "r" (b_zero_point)
        :
        "cc", "memory", "x4", "x5", "x6", "x7", "x8", "x9",
                "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
         //  std::cout << c[0] << std::endl;
    }
#endif


    void compute_quant_kernel_with_packed_input_a7w7(
            const int8_t* packed_input, // packed_input
            const int32_t* input_sums, // input_sums
            const int8_t* packed_weight, // packed weight
            const int32_t* weight_sums,
            int32_t* c, // output
            size_t kc, size_t ks, size_t c_stride,
            const int start_row, const int end_row,
            const int start_col, const int end_col,
            struct qconv_neon_params qconv_params
    ){
        const int8_t* input_ptr = packed_input;
        const int8_t* weight_base = packed_weight;
        const int8_t* weight_ptr = packed_weight;

        const int32_t* rhs_base_sums = weight_sums;
        int32_t* dst_point = c;

        const size_t stride_4 = c_stride * 4;
        const int32_t b_zero_point = qconv_params.kernel_zero_point;

        int col_pos = start_col;
        int row_pos = start_row;

        const size_t dst_addition = 4 * c_stride * sizeof(int32_t) - (end_col - start_col) * sizeof(int32_t) ;

        const size_t depth = kc * ks;

        asm volatile (
        // clang-format off
        // make v16 -- v31 be zero.
        // They will be replaced when tackling kernel zero.

#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

        // Clear the accumulators.
        MAKE_ZERO(v16)
        MAKE_ZERO(v17)
        MAKE_ZERO(v18)
        MAKE_ZERO(v19)
        MAKE_ZERO(v20)
        MAKE_ZERO(v21)
        MAKE_ZERO(v22)
        MAKE_ZERO(v23)
        MAKE_ZERO(v24)
        MAKE_ZERO(v25)
        MAKE_ZERO(v26)
        MAKE_ZERO(v27)
        MAKE_ZERO(v28)
        MAKE_ZERO(v29)
        MAKE_ZERO(v30)
        MAKE_ZERO(v31)

        "mov x1, #16\n"

        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        "1: \n"
        "cmp x1, %[depth]\n"
        "beq 79f\n"

        "2: \n"

        "sadalp  v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"

        "sadalp  v17.4s, v9.8h\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
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

        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"

        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"

        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"


        "sadalp  v24.4s, v8.8h\n"
        "smull  v8.8h,  v0.8b,  v4.8b\n"

        "sadalp  v25.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"

        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

        "sadalp  v26.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"

        "sadalp  v27.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"

        "sadalp  v28.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"

        "sadalp  v29.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"

        "sadalp  v30.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"

        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        "add x1, x1, #16\n"

        "cmp x1, %x[depth]\n"
        "blt 2b\n"

        "79: \n"

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

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
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

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 4x4 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x4 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally
        // It should be clear that output is row major.
        // The layout is HWC_o
        "addp v16.4s, v16.4s, v20.4s \n"
        "addp v17.4s, v17.4s, v21.4s \n"
        "addp v18.4s, v18.4s, v22.4s \n"
        "addp v19.4s, v19.4s, v23.4s \n"

        "addp v24.4s, v24.4s, v28.4s\n"
        "addp v25.4s, v25.4s, v29.4s\n"
        "addp v26.4s, v26.4s, v30.4s\n"
        "addp v27.4s, v27.4s, v31.4s\n"

        "addp v16.4s, v16.4s, v24.4s\n"
        "addp v17.4s, v17.4s, v25.4s\n"
        "addp v18.4s, v18.4s, v26.4s\n"
        "addp v19.4s, v19.4s, v27.4s\n"
        // If the end_col is processed,
        // then we will go to the next row;

     //  "add %w[col], %w[col], #4\n"
       "add %w[col], %w[col], #4\n"
       "cmp %w[col], %w[end_col]\n"
       "bge 4f\n"

       // If col < end_col, then deal with the next col.
       "add %[rhs_mov], %[rhs_mov], %[depth], lsl #2\n"
       "b 5f\n"

       "4: \n"
       // go back to the first col.
       // if the last col is processed,
       // we should come back to the base ptr.
       "mov %[rhs_mov], %[rhs_base_ptr]\n"

       // Now we need to advance to the next row.
       // If we already finished the last row, then in principle
       // we are done, however, we can't just return here, as we
       // need to allow the end work of the current block to complete.
       // The good news is that at this point it doesn't matter what data
       // we load for the next row, since we will exit from the main loop
       // below before actually storing anything computed from that data.
       "cmp %w[row], %w[end_row]\n"
       "bge 5f\n" // If yes, just carry on without updating the row pointer

       // If it is not the last row, we only needs to advance to next row.
       "add %[lhs_mov], %[lhs_mov], %[depth], lsl #2\n"


       "5: \n"
       "mov %[lhs_ptr], %[lhs_mov]\n"
       "mov %[rhs_ptr], %[rhs_mov]\n"
       // End work of current block.
       "ld1 {v21.4s}, [%[rhs_sum_ptr]], #16\n"

       "dup v20.4s, %w[zero_point] \n"
       "ld1 {v22.4s}, [%[lhs_sum_ptr]]\n"

       "dup v24.4s, v22.s[0]\n"
       "dup v25.4s, v22.s[1]\n"
       "dup v26.4s, v22.s[2]\n"
       "dup v27.4s, v22.s[3]\n"

       "add v16.4s, v16.4s, v21.4s\n"
       "add v17.4s, v17.4s, v21.4s\n"
       "add v18.4s, v18.4s, v21.4s\n"
       "add v19.4s, v19.4s, v21.4s\n"

       // Now that we know that LHS and RHS data the next iteration of the
       // main loop will need to load, we start loading the first 32 bytes of
       // each of LHS and RHS, int v0 -- v3
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

       "mls v16.4s, v20.4s, v24.4s\n"
       "mls v17.4s, v20.4s, v25.4s\n"
       "mls v18.4s, v20.4s, v26.4s\n"
       "mls v19.4s, v20.4s, v27.4s\n"

       MAKE_ZERO(v20)
       MAKE_ZERO(v21)
       MAKE_ZERO(v22)
       MAKE_ZERO(v23)
       MAKE_ZERO(v24)
       MAKE_ZERO(v25)
       MAKE_ZERO(v26)
       MAKE_ZERO(v27)
       MAKE_ZERO(v28)
       MAKE_ZERO(v29)
       MAKE_ZERO(v30)
       MAKE_ZERO(v31)


      "st1 {v16.4s}, [%[dst_point]]\n"
      "add x2, %[dst_point], %[stride_4]\n"
      "st1 {v17.4s}, [x2]\n"
      "add x2, x2, %[stride_4]\n"
      "st1 {v18.4s}, [x2]       \n"
      "add x2, x2, %[stride_4]\n"
      "st1 {v19.4s}, [x2]\n"

      // The destination of first row needs to advance 4 elements.
      "add %[dst_point], %[dst_point], #16\n"

       MAKE_ZERO(v16)
       MAKE_ZERO(v17)
       MAKE_ZERO(v18)
       MAKE_ZERO(v19)

       // for the next block: perform the first few multiply-adds on the data
       // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"
        // Move to the next block of destination matrix, for the next iter
        // of the main loop. Notice that lhs_ptr and rhs_ptr have already been updated.

        // If col == end_col.
        "cmp %w[col], %w[end_col]\n"
        "beq 20f\n"

        "b 21f\n"

        "20: \n"
        "mov %w[col], %w[start_col]\n" // mov back to the first col.
        "add %w[row], %w[row], #4\n" // mov to the next row

        "add %[lhs_sum_ptr], %[lhs_sum_ptr], #16\n"

        "mov %[rhs_sum_ptr], %[rhs_base_sums]\n"

        "add %[dst_point], %[dst_point], %[dst_addition]\n"

        "21: \n"

        // main loop exit condition: have we hit the end column
        "mov x1, #16\n"

        "cmp %w[row], %w[end_row]\n"
        "blt 1b\n"
#undef MAKE_ZERO
        :
        [lhs_ptr] "+r" (packed_input),
        [lhs_mov] "+r" (input_ptr),
        [lhs_sum_ptr] "+r" (input_sums),
        [rhs_ptr] "+r" (packed_weight),
        [rhs_mov] "+r" (weight_ptr),
        [rhs_sum_ptr] "+r" (weight_sums),
        [row] "+r" (row_pos),
        [col] "+r" (col_pos),
        [dst_point] "+r" (dst_point)
        :
        [depth] "r" (depth),
        [start_col] "r" (start_col),
        [end_row] "r" (end_row),
        [end_col] "r" (end_col),
        [zero_point] "r" (b_zero_point),
        [rhs_base_ptr] "r" (weight_base),
        [rhs_base_sums] "r" (rhs_base_sums),
        [stride_4] "r" (stride_4),
        [dst_addition] "r" (dst_addition)

     //   [dst_addition] "r" (dst_addition)
        :
        "cc", "memory", "x1", "x2",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");


    }




}
