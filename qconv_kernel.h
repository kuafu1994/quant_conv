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
} // quant_conv


#endif //QUANT_CONV_QCONV_KERNEL_H
