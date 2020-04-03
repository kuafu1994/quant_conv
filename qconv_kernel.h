//
// Created by PENGFEI ZHANG on 2020/3/26.
//

#ifndef QUANT_CONV_QCONV_KERNEL_H
#define QUANT_CONV_QCONV_KERNEL_H

#include <stddef.h>
#include <stdint.h>
#include "params.h"
#include "common.h"

namespace quant_conv {

#define STRING(s) STR_UNEXPANDED(s)
#define STR_UNEXPANDED(s) #s

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
    );

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
    );

    void compute_quant_kernel_with_packed_input_a7w7(
            size_t mr,
            size_t nr,
            size_t kc, // input_channels
            size_t ks, // kernel_size
            const int8_t* packed_input, // indirection_buffer
            const int32_t* input_sums,
            const void* w, // packed weight
            int32_t* c, // output
            size_t c_stride, // output_channels
            struct qconv_neon_params qconv_params
    );

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
    );

    struct kernel_params {

        int32_t start_row; // 0
        int32_t start_col; // 4
        int32_t end_row; // 8
        int32_t end_col; // 12
        int32_t kernel_zero_point; // 16
        int32_t input_zero_point; // 20

    };
} // quant_conv


#endif //QUANT_CONV_QCONV_KERNEL_H
