//
// Created by PENGFEI ZHANG on 2020/4/1.
//

#include "pack_kernel.h"

namespace quant_conv {
    void pack_8bit_neon_input(const int8_t **indirection_a, const int input_channel, const int kernel_size,
                              int8_t *packed_ptr, int32_t *sums_ptr) {


        // TODO: eliminate redundant computations, we can reduce (1 - 1 / ks) computation here.

        const int8_t ** a = indirection_a;
        int8_t* packed_w = packed_ptr;
        int32_t* p_sums = sums_ptr;

        asm volatile (
        // clang-format off

        // Load pointer from indirection_a
#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        MAKE_ZERO(v28)
        MAKE_ZERO(v29)
        MAKE_ZERO(v30)
        MAKE_ZERO(v31)

#undef MAKE_ZERO

        "mov w9, %w[ks]\n"

        "2: \n"

        "ldr x10, [%[p_a]], #8\n"
        "ldr x11, [%[p_a]], #8\n"
        "ldr x12, [%[p_a]], #8\n"
        "ldr x13, [%[p_a]], #8\n"

        "lsr w8, %w[kc], #4 \n"
        "subs w8, w8, #1\n"

        "ld1 {v0.16b}, [x10], #16\n"
        "ld1 {v1.16b}, [x11], #16\n"
        "ld1 {v2.16b}, [x12], #16\n"
        "ld1 {v3.16b}, [x13], #16\n"

        "beq 1f\n"

        "0: \n"
        "saddlp v24.8h, v0.16b\n"
        "st1 {v0.16b}, [%[packed_ptr]], #16\n"

        "saddlp v25.8h, v1.16b\n"
        "st1 {v1.16b}, [%[packed_ptr]], #16\n"

        "saddlp v26.8h, v2.16b\n"
        "st1  {v2.16b}, [%[packed_ptr]], #16\n"

        "saddlp v27.8h, v3.16b\n"
        "st1 {v3.16b}, [%[packed_ptr]], #16\n"

        "subs w8, w8, #1\n"

        "sadalp v28.4s, v24.8h\n"
        "ld1 {v0.16b}, [x10], #16\n"

        "sadalp v29.4s, v25.8h\n"
        "ld1 {v1.16b}, [x11], #16\n"

        "sadalp v30.4s, v26.8h\n"
        "ld1 {v2.16b}, [x12], #16\n"

        "sadalp v31.4s, v27.8h\n"
        "ld1 {v3.16b}, [x13], #16\n"

        "bne 0b\n"

        "1: \n"

        "saddlp v24.8h, v0.16b\n"
        "st1 {v0.16b},[%[packed_ptr]],#16\n"

        "saddlp v25.8h, v1.16b\n"
        "st1 {v1.16b},[%[packed_ptr]],#16\n"

        "saddlp v26.8h, v2.16b\n"
        "st1 {v2.16b},[%[packed_ptr]],#16\n"

        "saddlp v27.8h, v3.16b\n"
        "st1 {v3.16b},[%[packed_ptr]],#16\n"

        "sadalp v28.4s, v24.8h\n"
        "sadalp v29.4s, v25.8h\n"
        "sadalp v30.4s, v26.8h\n"
        "sadalp v31.4s, v27.8h\n"

        "subs w9, w9, #1\n"
        "bne 2b\n"

        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        "st1 {v28.4s}, [%[sums_ptr]], #16\n"
        :
        [p_a] "+r"(a),
        [packed_ptr] "+r"(packed_w),
        [sums_ptr] "+r"(p_sums)
        :
        [kc] "r"(input_channel),
        [ks] "r"(kernel_size)
        :
        "cc", "memory", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
                "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v15", "v17",
                "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                "v27", "v28", "v29", "v30", "v31"
        );
    }

    void pack_8bit_neon_weight(const int8_t *w, const int input_channel, const int kernel_size,
                              int8_t *packed_ptr, int32_t *sums_ptr, const int8_t input_zero_point, const int8_t kernel_zero_point) {




        const int32_t off = input_zero_point * kernel_zero_point * kernel_size * input_channel;
        const int32_t izp = input_zero_point;
        int8_t* packed_w = packed_ptr;

        const int stride = kernel_size * input_channel;
        const int8_t* w0 = w;
        const int8_t* w1 = w0 + stride;
        const int8_t* w2 = w1 + stride;
        const int8_t* w3 = w2 + stride;
        asm volatile (
        // clang-format off

        "dup v16.4s, %w[off]\n"
        "dup v17.4s, %w[izp]\n"
        // Load pointer from indirection_a
#define MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        MAKE_ZERO(v28)
        MAKE_ZERO(v29)
        MAKE_ZERO(v30)
        MAKE_ZERO(v31)
#undef MAKE_ZERO
        "mov w9, %w[ks]\n"

        "2: \n"

        "ld1 {v0.16b}, [%[w0]], #16\n"
        "ld1 {v1.16b}, [%[w1]], #16\n"

        "lsr w8, %w[kc], #4 \n" // w8 = kc >> 4
        "subs w8, w8, #1\n" // w8 = w8 - 1

        "ld1 {v2.16b}, [%[w2]], #16\n"
        "ld1 {v3.16b}, [%[w3]], #16\n"

        "beq 1f\n"

        "0: \n"
        "saddlp v24.8h, v0.16b\n"
        "st1 {v0.16b}, [%[pp]], #16\n"

        "saddlp v25.8h, v1.16b\n"
        "st1 {v1.16b}, [%[pp]], #16\n"

        "saddlp v26.8h, v2.16b\n"
        "st1 {v2.16b}, [%[pp]], #16\n"

        "saddlp v27.8h, v3.16b\n"
        "st1 {v3.16b}, [%[pp]], #16\n"

        "subs w8, w8, #1\n"

        "sadalp v28.4s, v24.8h\n"
        "ld1 {v0.16b}, [%[w0]], #16\n"
        "sadalp v29.4s, v25.8h\n"
        "ld1 {v1.16b}, [%[w1]], #16\n"
        "sadalp v30.4s, v26.8h\n"
        "ld1 {v2.16b}, [%[w2]], #16\n"
        "sadalp v31.4s, v27.8h\n"
        "ld1 {v3.16b}, [%[w3]], #16\n"

        "bne 0b\n"

        "1: \n"
        "saddlp v24.8h, v0.16b\n"
        "st1 {v0.16b}, [%[pp]], #16\n"

        "saddlp v25.8h, v1.16b\n"
        "st1 {v1.16b}, [%[pp]], #16\n"

        "saddlp v26.8h, v2.16b\n"
        "st1 {v2.16b}, [%[pp]], #16\n"

        "saddlp v27.8h, v3.16b\n"
        "st1 {v3.16b}, [%[pp]], #16\n"

        "sadalp v28.4s, v24.8h\n"
        "sadalp v29.4s, v25.8h\n"
        "sadalp v30.4s, v26.8h\n"
        "sadalp v31.4s, v27.8h\n"

        "subs w9, w9, #1\n"
        "bne 2b\n"

        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        "mul v28.4s, v28.4s, v17.4s\n"
        "sub v16.4s, v16.4s, v28.4s\n"

        "st1 {v16.4s}, [%[sums_ptr]], #16\n"
        :
        [w0] "+r"(w0),
        [w1] "+r" (w1),
        [w2] "+r" (w2),
        [w3] "+r" (w3),
        [pp] "+r"(packed_w),
        [sums_ptr] "+r"(sums_ptr)
        :
        [kc] "r"(input_channel),
        [ks] "r"(kernel_size),
        [off] "r"(off),
        [izp] "r" (izp)
        :
        "cc", "memory", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
                "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v15", "v17",
                "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                "v27", "v28", "v29", "v30", "v31"
        );
    }


}