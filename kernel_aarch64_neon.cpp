//
// Created by PENGFEI ZHANG on 2020/3/26.
//

#include <stdint.h>
#include "params.h"
#include <iostream>
#include <string.h>
#include <assert.h>
#include "qconv_kernel.h"

namespace quant_conv {


    void compute_quant_kernel_with_packed_input_a7w7(
            const int8_t* packed_input, // packed_input
            const int32_t* input_sums, // input_sums
            const int8_t* packed_weight, // packed weight
            const int32_t* weight_sums,
            int32_t* c, // output
            size_t kc, // input_channels
            size_t ks, // kernel_size
            size_t c_stride, // output_channels
            const int start_row,
            const int end_row,
            const int start_col,
            const int end_col,
            struct qconv_neon_params qconv_params
    ){
        assert(((end_row - start_row) % 4) == 0);
        assert(((end_col - start_col) % 4) == 0);

        const int32_t b_zero_point = qconv_params.kernel_zero_point;
        const size_t depth = kc * ks;

        for(int ii = 0; ii < end_row - start_row ; ii += 4) {

            const int8_t* weight_ptr = packed_weight;
            const int32_t* rhs_sums_ptr = weight_sums;
            int col_pos = start_col;

            const int8_t* input_ptr = packed_input + depth * ii;
            //const int8_t* const input_base_ptr = input_ptr;
            const int32_t* input_sums_ptr = input_sums + ii;
            int32_t* c0 = c + ii * c_stride;
            int32_t* c1 = c0 + c_stride;
            int32_t* c2 = c1 + c_stride;
            int32_t* c3 = c2 + c_stride;


            asm volatile (

#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

            "1: \n"

            // Load the first 64 bytes of LHS and RHS data.
            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v0.16b, v1.16b}, [%[lhs_ptr]], #32\n"
           // "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"
          //  "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"


            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v4.16b, v5.16b}, [%[rhs_ptr]], #32\n"
            //"ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
            "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"
            //"ld1 {v7.16b}, [%[rhs_ptr]], #16\n"


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


            "cmp x1, %[depth]\n"
            "beq 79f\n"

            "2: \n"

            "prfm pldl1keep, [%[rhs_ptr],#512]\n"
            "ld1 {v4.16b, v5.16b}, [%[rhs_ptr]], #32\n"
            //"ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
          //  "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
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

           // "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"


            "smlal2   v12.8h,  v0.16b,  v7.16b\n"
           //"ld1 {v0.16b}, [%[lhs_ptr]], #16\n"

            "smlal2   v13.8h,  v1.16b,  v7.16b\n"
          //  "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

            "smlal2   v14.8h,  v2.16b,  v7.16b\n"
           // "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"

            "smlal2   v15.8h,  v3.16b,  v7.16b\n"
           // "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"


            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v0.16b, v1.16b}, [%[lhs_ptr]], #32\n"
           // "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"
           // "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

            "sadalp  v24.4s, v8.8h\n"
            "smull  v8.8h,  v0.8b,  v4.8b\n"

            "sadalp  v25.4s, v9.8h\n"
            "smull    v9.8h,  v1.8b,  v4.8b\n"


          //  "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

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

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"

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

            // Return back
            "sub %[lhs_ptr], %[lhs_ptr], %[depth], lsl #2\n"

            "add %w[col], %w[col], #4\n"

            // End work of current block.
            "ld1 {v21.4s}, [%[rhs_sums_ptr]], #16\n"

            "dup v20.4s, %w[zero_point] \n"
            "ld1 {v22.4s}, [%[lhs_sums_ptr]]\n"

            "dup v24.4s, v22.s[0]\n"
            "dup v25.4s, v22.s[1]\n"
            "dup v26.4s, v22.s[2]\n"
            "dup v27.4s, v22.s[3]\n"

            "add v16.4s, v16.4s, v21.4s\n"
            "add v17.4s, v17.4s, v21.4s\n"
            "add v18.4s, v18.4s, v21.4s\n"
            "add v19.4s, v19.4s, v21.4s\n"

            "mls v16.4s, v20.4s, v24.4s\n"
            "mls v17.4s, v20.4s, v25.4s\n"
            "mls v18.4s, v20.4s, v26.4s\n"
            "mls v19.4s, v20.4s, v27.4s\n"

            "cmp %w[col], %w[end_col]\n"

            "st1 {v16.4s}, [%[p_c0]], #16\n"
            "st1 {v17.4s}, [%[p_c1]], #16\n"
            "st1 {v18.4s}, [%[p_c2]], #16\n"
            "st1 {v19.4s}, [%[p_c3]], #16\n"

            "blt 1b\n"
#undef MAKE_ZERO
            :
            [lhs_ptr] "+r"(input_ptr),
            [rhs_ptr] "+r"(weight_ptr),
            [rhs_sums_ptr] "+r"(rhs_sums_ptr),
            [col] "+r"(col_pos),
            [p_c0] "+r" (c0),
            [p_c1] "+r" (c1),
            [p_c2] "+r" (c2),
            [p_c3] "+r" (c3)
            :
            [depth] "r" (depth),
            [end_col] "r"(end_col),
            [zero_point] "r"(b_zero_point),
            [lhs_sums_ptr] "r"(input_sums_ptr)
        //    [lhs_base_ptr] "r" (input_base_ptr)
            :
            "cc", "memory", "x1", "x3",
                    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

        }
    }

    void compute_quant_kernel_with_packed_input_a2w2(
            const int8_t* packed_input, // packed_input
            const int32_t* input_sums, // input_sums
            const int8_t* packed_weight, // packed weight
            const int32_t* weight_sums,
            int32_t* c, // output
            size_t kc, // input_channels
            size_t ks, // kernel_size
            size_t c_stride, // output_channels
            const int start_row,
            const int end_row,
            const int start_col,
            const int end_col,
            struct qconv_neon_params qconv_params
    ) {

        assert(((end_row - start_row) % 4) == 0);
        assert(((end_col - start_col) % 4) == 0);

        const int32_t b_zero_point = qconv_params.kernel_zero_point;
        const size_t depth = kc * ks;

        for(int ii = 0; ii < end_row - start_row ; ii += 4) {

            const int8_t* weight_ptr = packed_weight;
            const int32_t* rhs_sums_ptr = weight_sums;
            int col_pos = start_col;

            const int8_t* input_ptr = packed_input + depth * ii;
            //const int8_t* const input_base_ptr = input_ptr;
            const int32_t* input_sums_ptr = input_sums + ii;
            int32_t* c0 = c + ii * c_stride;
            int32_t* c1 = c0 + c_stride;
            int32_t* c2 = c1 + c_stride;
            int32_t* c3 = c2 + c_stride;


            asm volatile (

#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

            "1: \n"

            // Load the first 64 bytes of LHS and RHS data.
            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v0.16b, v1.16b}, [%[lhs_ptr]], #32\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v4.16b, v5.16b}, [%[rhs_ptr]], #32\n"
            "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"

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


            "mov x1, #32\n"
            "cmp x1, %[depth]\n"
            "beq 79f\n"

            "2: \n"
            "smlal v16.8h, v0.8b, v4.8b\n"
            "smlal v17.8h, v1.8b, v4.8b\n"
            "smlal v18.8h, v2.8b, v4.8b\n"
            "smlal v19.8h, v3.8b, v4.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v8.16b}, [%[lhs_ptr]], #16\n"

            "smlal v20.8h, v0.8b, v5.8b\n"
            "smlal v21.8h, v1.8b, v5.8b\n"
            "smlal v22.8h, v2.8b, v5.8b\n"
            "smlal v23.8h, v3.8b, v5.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v9.16b}, [%[lhs_ptr]], #16\n"

            "smlal v24.8h, v0.8b, v6.8b\n"
            "smlal v25.8h, v1.8b, v6.8b\n"
            "smlal v26.8h, v2.8b, v6.8b\n"
            "smlal v27.8h, v3.8b, v6.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v10.16b}, [%[lhs_ptr]], #16\n"

            "smlal v28.8h, v0.8b, v7.8b\n"
            "smlal v29.8h, v1.8b, v7.8b\n"
            "smlal v30.8h, v2.8b, v7.8b\n"
            "smlal v31.8h, v3.8b, v7.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v11.16b}, [%[lhs_ptr]], #16\n"

            "smlal2 v16.8h, v0.16b, v4.16b\n"
            "smlal2 v17.8h, v1.16b, v4.16b\n"
            "smlal2 v18.8h, v2.16b, v4.16b\n"
            "smlal2 v19.8h, v3.16b, v4.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v12.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v20.8h, v0.16b, v5.16b\n"
            "smlal2 v21.8h, v1.16b, v5.16b\n"
            "smlal2 v22.8h, v2.16b, v5.16b\n"
            "smlal2 v23.8h, v3.16b, v5.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v13.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v24.8h, v0.16b, v6.16b\n"
            "smlal2 v25.8h, v1.16b, v6.16b\n"
            "smlal2 v26.8h, v2.16b, v6.16b\n"
            "smlal2 v27.8h, v3.16b, v6.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v14.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v28.8h, v0.16b, v7.16b\n"
            "smlal2 v29.8h, v1.16b, v7.16b\n"
            "smlal2 v30.8h, v2.16b, v7.16b\n"
            "smlal2 v31.8h, v3.16b, v7.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v15.16b}, [%[rhs_ptr]], #16\n"

            "smlal v16.8h, v8.8b, v12.8b\n"
            "smlal v17.8h, v9.8b, v12.8b\n"
            "smlal v18.8h, v10.8b, v12.8b\n"
            "smlal v19.8h, v11.8b, v12.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"


            "smlal v20.8h, v8.8b, v13.8b\n"
            "smlal v21.8h, v9.8b, v13.8b\n"
            "smlal v22.8h, v10.8b, v13.8b\n"
            "smlal v23.8h, v11.8b, v13.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"


            "smlal v24.8h, v8.8b, v14.8b\n"
            "smlal v25.8h, v9.8b, v14.8b\n"
            "smlal v26.8h, v10.8b, v14.8b\n"
            "smlal v27.8h, v11.8b, v14.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"

            "smlal v28.8h, v8.8b, v15.8b\n"
            "smlal v29.8h, v9.8b, v15.8b\n"
            "smlal v30.8h, v10.8b, v15.8b\n"
            "smlal v31.8h, v11.8b, v15.8b\n"


            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

            "add x1, x1, #32\n"
            "cmp x1, %[depth]\n"

            "smlal2 v16.8h, v8.16b, v12.16b\n"
            "smlal2 v17.8h, v9.16b, v12.16b\n"
            "smlal2 v18.8h, v10.16b, v12.16b\n"
            "smlal2 v19.8h, v11.16b, v12.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v20.8h, v8.16b, v13.16b\n"
            "smlal2 v21.8h, v9.16b, v13.16b\n"
            "smlal2 v22.8h, v10.16b, v13.16b\n"
            "smlal2 v23.8h, v11.16b, v13.16b\n"


            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v24.8h, v8.16b, v14.16b\n"
            "smlal2 v25.8h, v9.16b, v14.16b\n"
            "smlal2 v26.8h, v10.16b, v14.16b\n"
            "smlal2 v27.8h, v11.16b, v14.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"


            "smlal2 v28.8h, v8.16b, v15.16b\n"
            "smlal2 v29.8h, v9.16b, v15.16b\n"
            "smlal2 v30.8h, v10.16b, v15.16b\n"
            "smlal2 v31.8h, v11.16b, v15.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"


            "blt 2b\n"

            "79: \n"

            "smlal v16.8h, v0.8b, v4.8b\n"
            "smlal v17.8h, v1.8b, v4.8b\n"
            "smlal v18.8h, v2.8b, v4.8b\n"
            "smlal v19.8h, v3.8b, v4.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v8.16b}, [%[lhs_ptr]], #16\n"

            "smlal v20.8h, v0.8b, v5.8b\n"
            "smlal v21.8h, v1.8b, v5.8b\n"
            "smlal v22.8h, v2.8b, v5.8b\n"
            "smlal v23.8h, v3.8b, v5.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v9.16b}, [%[lhs_ptr]], #16\n"

            "smlal v24.8h, v0.8b, v6.8b\n"
            "smlal v25.8h, v1.8b, v6.8b\n"
            "smlal v26.8h, v2.8b, v6.8b\n"
            "smlal v27.8h, v3.8b, v6.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v10.16b}, [%[lhs_ptr]], #16\n"

            "smlal v28.8h, v0.8b, v7.8b\n"
            "smlal v29.8h, v1.8b, v7.8b\n"
            "smlal v30.8h, v2.8b, v7.8b\n"
            "smlal v31.8h, v3.8b, v7.8b\n"

            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v11.16b}, [%[lhs_ptr]], #16\n"

            "smlal2 v16.8h, v0.16b, v4.16b\n"
            "smlal2 v17.8h, v1.16b, v4.16b\n"
            "smlal2 v18.8h, v2.16b, v4.16b\n"
            "smlal2 v19.8h, v3.16b, v4.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v12.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v20.8h, v0.16b, v5.16b\n"
            "smlal2 v21.8h, v1.16b, v5.16b\n"
            "smlal2 v22.8h, v2.16b, v5.16b\n"
            "smlal2 v23.8h, v3.16b, v5.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v13.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v24.8h, v0.16b, v6.16b\n"
            "smlal2 v25.8h, v1.16b, v6.16b\n"
            "smlal2 v26.8h, v2.16b, v6.16b\n"
            "smlal2 v27.8h, v3.16b, v6.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v14.16b}, [%[rhs_ptr]], #16\n"

            "smlal2 v28.8h, v0.16b, v7.16b\n"
            "smlal2 v29.8h, v1.16b, v7.16b\n"
            "smlal2 v30.8h, v2.16b, v7.16b\n"
            "smlal2 v31.8h, v3.16b, v7.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v15.16b}, [%[rhs_ptr]], #16\n"

            "smlal v16.8h, v8.8b, v12.8b\n"
            "smlal v17.8h, v9.8b, v12.8b\n"
            "smlal v18.8h, v10.8b, v12.8b\n"
            "smlal v19.8h, v11.8b, v12.8b\n"

            "smlal v20.8h, v8.8b, v13.8b\n"
            "smlal v21.8h, v9.8b, v13.8b\n"
            "smlal v22.8h, v10.8b, v13.8b\n"
            "smlal v23.8h, v11.8b, v13.8b\n"

            "smlal v24.8h, v8.8b, v14.8b\n"
            "smlal v25.8h, v9.8b, v14.8b\n"
            "smlal v26.8h, v10.8b, v14.8b\n"
            "smlal v27.8h, v11.8b, v14.8b\n"

            "smlal v28.8h, v8.8b, v15.8b\n"
            "smlal v29.8h, v9.8b, v15.8b\n"
            "smlal v30.8h, v10.8b, v15.8b\n"
            "smlal v31.8h, v11.8b, v15.8b\n"

            "smlal2 v16.8h, v8.16b, v12.16b\n"
            "smlal2 v17.8h, v9.16b, v12.16b\n"
            "smlal2 v18.8h, v10.16b, v12.16b\n"
            "smlal2 v19.8h, v11.16b, v12.16b\n"

            "smlal2 v20.8h, v8.16b, v13.16b\n"
            "smlal2 v21.8h, v9.16b, v13.16b\n"
            "smlal2 v22.8h, v10.16b, v13.16b\n"
            "smlal2 v23.8h, v11.16b, v13.16b\n"

            "smlal2 v24.8h, v8.16b, v14.16b\n"
            "smlal2 v25.8h, v9.16b, v14.16b\n"
            "smlal2 v26.8h, v10.16b, v14.16b\n"
            "smlal2 v27.8h, v11.16b, v14.16b\n"

            "smlal2 v28.8h, v8.16b, v15.16b\n"
            "smlal2 v29.8h, v9.16b, v15.16b\n"
            "smlal2 v30.8h, v10.16b, v15.16b\n"
            "smlal2 v31.8h, v11.16b, v15.16b\n"

            "saddlp v16.4s, v16.8h\n"
            "saddlp v17.4s, v17.8h\n"
            "saddlp v18.4s, v18.8h\n"
            "saddlp v19.4s, v19.8h\n"
            "saddlp v20.4s, v20.8h\n"
            "saddlp v21.4s, v21.8h\n"
            "saddlp v22.4s, v22.8h\n"
            "saddlp v23.4s, v23.8h\n"
            "saddlp v24.4s, v24.8h\n"
            "saddlp v25.4s, v25.8h\n"
            "saddlp v26.4s, v26.8h\n"
            "saddlp v27.4s, v27.8h\n"
            "saddlp v28.4s, v28.8h\n"
            "saddlp v29.4s, v29.8h\n"
            "saddlp v30.4s, v30.8h\n"
            "saddlp v31.4s, v31.8h\n"

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

#if 0
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


            "cmp x1, %[depth]\n"
            "beq 79f\n"

            "2: \n"

            "prfm pldl1keep, [%[rhs_ptr],#512]\n"
            "ld1 {v4.16b, v5.16b}, [%[rhs_ptr]], #32\n"
            //"ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
            //  "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
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

            // "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"


            "smlal2   v12.8h,  v0.16b,  v7.16b\n"
            //"ld1 {v0.16b}, [%[lhs_ptr]], #16\n"

            "smlal2   v13.8h,  v1.16b,  v7.16b\n"
            //  "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

            "smlal2   v14.8h,  v2.16b,  v7.16b\n"
            // "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"

            "smlal2   v15.8h,  v3.16b,  v7.16b\n"
            // "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"


            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v0.16b, v1.16b}, [%[lhs_ptr]], #32\n"
            // "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"
            // "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

            "sadalp  v24.4s, v8.8h\n"
            "smull  v8.8h,  v0.8b,  v4.8b\n"

            "sadalp  v25.4s, v9.8h\n"
            "smull    v9.8h,  v1.8b,  v4.8b\n"


            //  "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

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

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"

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
#endif
            // Return back
            "sub %[lhs_ptr], %[lhs_ptr], %[depth], lsl #2\n"

            "add %w[col], %w[col], #4\n"

            // End work of current block.
            "ld1 {v21.4s}, [%[rhs_sums_ptr]], #16\n"

            "dup v20.4s, %w[zero_point] \n"
            "ld1 {v22.4s}, [%[lhs_sums_ptr]]\n"

            "dup v24.4s, v22.s[0]\n"
            "dup v25.4s, v22.s[1]\n"
            "dup v26.4s, v22.s[2]\n"
            "dup v27.4s, v22.s[3]\n"

            "add v16.4s, v16.4s, v21.4s\n"
            "add v17.4s, v17.4s, v21.4s\n"
            "add v18.4s, v18.4s, v21.4s\n"
            "add v19.4s, v19.4s, v21.4s\n"

            "mls v16.4s, v20.4s, v24.4s\n"
            "mls v17.4s, v20.4s, v25.4s\n"
            "mls v18.4s, v20.4s, v26.4s\n"
            "mls v19.4s, v20.4s, v27.4s\n"

            "cmp %w[col], %w[end_col]\n"

            "st1 {v16.4s}, [%[p_c0]], #16\n"
            "st1 {v17.4s}, [%[p_c1]], #16\n"
            "st1 {v18.4s}, [%[p_c2]], #16\n"
            "st1 {v19.4s}, [%[p_c3]], #16\n"

            "blt 1b\n"
#undef MAKE_ZERO
            :
            [lhs_ptr] "+r"(input_ptr),
            [rhs_ptr] "+r"(weight_ptr),
            [rhs_sums_ptr] "+r"(rhs_sums_ptr),
            [col] "+r"(col_pos),
            [p_c0] "+r" (c0),
            [p_c1] "+r" (c1),
            [p_c2] "+r" (c2),
            [p_c3] "+r" (c3)
            :
            [depth] "r" (depth),
            [end_col] "r"(end_col),
            [zero_point] "r"(b_zero_point),
            [lhs_sums_ptr] "r"(input_sums_ptr)
            //    [lhs_base_ptr] "r" (input_base_ptr)
            :
            "cc", "memory", "x1", "x3",
                    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

        }


    }
    void compute_quant_kernel_with_packed_input_sdot(
            const int8_t* packed_input, // packed_input
            const int32_t* input_sums, // input_sums
            const int8_t* packed_weight, // packed weight
            const int32_t* weight_sums,
            int32_t* c, // output
            size_t kc, // input_channels
            size_t ks, // kernel_size
            size_t c_stride, // output_channels
            const int start_row,
            const int end_row,
            const int start_col,
            const int end_col,
            struct qconv_neon_params qconv_params
    ) {
        assert(((end_row - start_row) % 4) == 0);
        assert(((end_col - start_col) % 4) == 0);

        const int32_t b_zero_point = qconv_params.kernel_zero_point;
        const size_t depth = kc * ks;

        for (int ii = 0; ii < end_row - start_row; ii += 4) {

            const int8_t *weight_ptr = packed_weight;
            const int32_t *rhs_sums_ptr = weight_sums;
            int col_pos = start_col;

            const int8_t *input_ptr = packed_input + depth * ii;
            //const int8_t* const input_base_ptr = input_ptr;
            const int32_t *input_sums_ptr = input_sums + ii;
            int32_t *c0 = c + ii * c_stride;
            int32_t *c1 = c0 + c_stride;
            int32_t *c2 = c1 + c_stride;
            int32_t *c3 = c2 + c_stride;


            asm volatile (
#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

            "1: \n"

            // Load the first 64 bytes of LHS and RHS data.
            "prfm pldl1keep, [%[lhs_ptr],#512]\n"
            "ld1 {v0.16b, v1.16b}, [%[lhs_ptr]], #32\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v4.16b, v5.16b}, [%[rhs_ptr]], #32\n"
            "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"


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


            "mov x1, #32\n"
            "cmp x1, %[depth]\n"
            "beq 79f\n"

            "2:\n"
            ".word 0x4e849410 // sdot v16.4s, v0.16b, v4.16b\n" // 0100 0001 0000
            ".word 0x4e849431 // sdot v17.4s, v1.16b, v4.16b\n" // 0100 0011 0001
            ".word 0x4e849452 // sdot v18.4s, v2.16b, v4.16b\n" // 0100 0101 0010
            ".word 0x4e849473 // sdot v19.4s, v3.16b, v4.16b\n" // 0100 0111 0011

            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v8.16b, v9.16b}, [%[lhs_ptr]], #32\n"
            "ld1 {v10.16b, v11.16b}, [%[lhs_ptr]], #32\n"

            ".word 0x4e859414 // sdot v20.4s, v0.16b, v5.16b\n" // 0100 0001 0100
            ".word 0x4e859435 // sdot v21.4s, v1.16b, v5.16b\n" // 0100 0011 0101
            ".word 0x4e859456 // sdot v22.4s, v2.16b, v5.16b\n"
            ".word 0x4e859477 // sdot v23.4s, v3.16b, v5.16b\n"

            #if 0
            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v10.16b, v11.16b}, [%[lhs_ptr]], #32\n"
            #endif
            ".word 0x4e869418 // sdot v24.4s, v0.16b, v6.16b\n" // 0100 0001 1000
            ".word 0x4e869439 // sdot v25.4s, v1.16b, v6.16b\n"
            ".word 0x4e86945a // sdot v26.4s, v2.16b, v6.16b\n"
            ".word 0x4e86947b // sdot v27.4s, v3.16b, v6.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v12.16b, v13.16b}, [%[rhs_ptr]], #32\n"
            "ld1 {v14.16b, v15.16b}, [%[rhs_ptr]], #32\n"

            ".word 0x4e87941c // sdot v28.4s, v0.16b, v7.16b\n"  //
            ".word 0x4e87943d // sdot v29.4s, v1.16b, v7.16b\n"
            ".word 0x4e87945e // sdot v30.4s, v2.16b, v7.16b\n"
            ".word 0x4e87947f // sdot v31.4s, v3.16b, v7.16b\n"
#if 0
            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v14.16b, v15.16b}, [%[rhs_ptr]], #32\n"
#endif
            ".word 0x4e8c9510 // sdot v16.4s, v8.16b, v12.16b\n" // 0101 0001 0000
            ".word 0x4e8c9531 // sdot v17.4s, v9.16b, v12.16b\n" //
            ".word 0x4e8c9552 // sdot v18.4s, v10.16b, v12.16b\n"
            ".word 0x4e8c9573 // sdot v19.4s, v11.16b, v12.16b\n"

            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v0.16b, v1.16b}, [%[lhs_ptr]], #32\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"

            ".word 0x4e8d9514 // sdot v20.4s, v8.16b, v13.16b\n"
            ".word 0x4e8d9535 // sdot v21.4s, v9.16b, v13.16b\n"
            ".word 0x4e8d9556 // sdot v22.4s, v10.16b, v13.16b\n"
            ".word 0x4e8d9577 // sdot v23.4s, v11.16b, v13.16b\n"
#if 0
            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v2.16b, v3.16b}, [%[lhs_ptr]], #32\n"
#endif
            "add x1, x1, #32\n"
            "cmp x1, %[depth]\n"

            ".word 0x4e8e9518 // sdot v24.4s, v8.16b, v14.16b\n"
            ".word 0x4e8e9539 // sdot v25.4s, v9.16b, v14.16b\n"
            ".word 0x4e8e955a // sdot v26.4s, v10.16b, v14.16b\n"
            ".word 0x4e8e957b // sdot v27.4s, v11.16b, v14.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v4.16b, v5.16b}, [%[rhs_ptr]], #32\n"
            "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"

            ".word 0x4e8f951c // sdot v28.4s, v8.16b, v15.16b\n"
            ".word 0x4e8f953d // sdot v29.4s, v9.16b, v15.16b\n"
            ".word 0x4e8f955e // sdot v30.4s, v10.16b, v15.16b\n"
            ".word 0x4e8f957f // sdot v31.4s, v11.16b, v15.16b\n"
#if 0
           "prfm pldl1keep, [%[rhs_ptr], #512]\n"
           "ld1 {v6.16b, v7.16b}, [%[rhs_ptr]], #32\n"
#endif
            "blt 2b\n"

            "79: \n"

            ".word 0x4e849410 // sdot v16.4s, v0.16b, v4.16b\n" // 0100 0001 0000
            ".word 0x4e849431 // sdot v17.4s, v1.16b, v4.16b\n" // 0100 0011 0001
            ".word 0x4e849452 // sdot v18.4s, v2.16b, v4.16b\n" // 0100 0101 0010
            ".word 0x4e849473 // sdot v19.4s, v3.16b, v4.16b\n" // 0100 0111 0011

            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v8.16b, v9.16b}, [%[lhs_ptr]], #32\n"
            "ld1 {v10.16b, v11.16b}, [%[lhs_ptr]], #32\n"

            ".word 0x4e859414 // sdot v20.4s, v0.16b, v5.16b\n" // 0100 0001 0100
            ".word 0x4e859435 // sdot v21.4s, v1.16b, v5.16b\n" // 0100 0011 0101
            ".word 0x4e859456 // sdot v22.4s, v2.16b, v5.16b\n"
            ".word 0x4e859477 // sdot v23.4s, v3.16b, v5.16b\n"
#if 0
            "prfm pldl1keep, [%[lhs_ptr], #512]\n"
            "ld1 {v10.16b, v11.16b}, [%[lhs_ptr]], #32\n"
#endif
            ".word 0x4e869418 // sdot v24.4s, v0.16b, v6.16b\n" // 0100 0001 1000
            ".word 0x4e869439 // sdot v25.4s, v1.16b, v6.16b\n"
            ".word 0x4e86945a // sdot v26.4s, v2.16b, v6.16b\n"
            ".word 0x4e86947b // sdot v27.4s, v3.16b, v6.16b\n"

            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v12.16b, v13.16b}, [%[rhs_ptr]], #32\n"
            "ld1 {v14.16b, v15.16b}, [%[rhs_ptr]], #32\n"

            ".word 0x4e87941c // sdot v28.4s, v0.16b, v7.16b\n"  //
            ".word 0x4e87943d // sdot v29.4s, v1.16b, v7.16b\n"
            ".word 0x4e87945e // sdot v30.4s, v2.16b, v7.16b\n"
            ".word 0x4e87947f // sdot v31.4s, v3.16b, v7.16b\n"
#if 0
            "prfm pldl1keep, [%[rhs_ptr], #512]\n"
            "ld1 {v14.16b, v15.16b}, [%[rhs_ptr]], #32\n"
#endif
            ".word 0x4e8c9510 // sdot v16.4s, v8.16b, v12.16b\n" // 0101 0001 0000
            ".word 0x4e8c9531 // sdot v17.4s, v9.16b, v12.16b\n" //
            ".word 0x4e8c9552 // sdot v18.4s, v10.16b, v12.16b\n"
            ".word 0x4e8c9573 // sdot v19.4s, v11.16b, v12.16b\n"

            ".word 0x4e8d9514 // sdot v20.4s, v8.16b, v13.16b\n"
            ".word 0x4e8d9535 // sdot v21.4s, v9.16b, v13.16b\n"
            ".word 0x4e8d9556 // sdot v22.4s, v10.16b, v13.16b\n"
            ".word 0x4e8d9577 // sdot v23.4s, v11.16b, v13.16b\n"

            ".word 0x4e8e9518 // sdot v24.4s, v8.16b, v14.16b\n"
            ".word 0x4e8e9539 // sdot v25.4s, v9.16b, v14.16b\n"
            ".word 0x4e8e955a // sdot v26.4s, v10.16b, v14.16b\n"
            ".word 0x4e8e957b // sdot v27.4s, v11.16b, v14.16b\n"

            ".word 0x4e8f951c // sdot v28.4s, v8.16b, v15.16b\n"
            ".word 0x4e8f953d // sdot v29.4s, v9.16b, v15.16b\n"
            ".word 0x4e8f955e // sdot v30.4s, v10.16b, v15.16b\n"
            ".word 0x4e8f957f // sdot v31.4s, v11.16b, v15.16b\n"

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

            "sub %[lhs_ptr], %[lhs_ptr], %[depth], lsl #2\n"
            "add %w[col], %w[col], #4\n"

            // End work of current block.
            "ld1 {v21.4s}, [%[rhs_sums_ptr]], #16\n"

            "dup v20.4s, %w[zero_point] \n"
            "ld1 {v22.4s}, [%[lhs_sums_ptr]]\n"

            "dup v24.4s, v22.s[0]\n"
            "dup v25.4s, v22.s[1]\n"
            "dup v26.4s, v22.s[2]\n"
            "dup v27.4s, v22.s[3]\n"

            "add v16.4s, v16.4s, v21.4s\n"
            "add v17.4s, v17.4s, v21.4s\n"
            "add v18.4s, v18.4s, v21.4s\n"
            "add v19.4s, v19.4s, v21.4s\n"

            "mls v16.4s, v20.4s, v24.4s\n"
            "mls v17.4s, v20.4s, v25.4s\n"
            "mls v18.4s, v20.4s, v26.4s\n"
            "mls v19.4s, v20.4s, v27.4s\n"

            "cmp %w[col], %w[end_col]\n"

            "st1 {v16.4s}, [%[p_c0]], #16\n"
            "st1 {v17.4s}, [%[p_c1]], #16\n"
            "st1 {v18.4s}, [%[p_c2]], #16\n"
            "st1 {v19.4s}, [%[p_c3]], #16\n"

            "blt 1b\n"
            :
            [lhs_ptr] "+r"(input_ptr),
            [rhs_ptr] "+r"(weight_ptr),
            [rhs_sums_ptr] "+r"(rhs_sums_ptr),
            [col] "+r"(col_pos),
            [p_c0] "+r"(c0),
            [p_c1] "+r"(c1),
            [p_c2] "+r"(c2),
            [p_c3] "+r"(c3)
            :
            [depth] "r"(depth),
            [end_col] "r"(end_col),
            [zero_point] "r"(b_zero_point),
            [lhs_sums_ptr] "r"(input_sums_ptr)
            //    [lhs_base_ptr] "r" (input_base_ptr)
            :
            "cc", "memory", "x1", "x3",
                    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
        }

    }


}
