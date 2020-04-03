//
// Created by PENGFEI ZHANG on 2020/4/1.
//

#ifndef QUANT_CONV_PACK_KERNEL_H
#define QUANT_CONV_PACK_KERNEL_H

#include <stdint.h>

namespace quant_conv {

    void pack_8bit_neon_input(const int8_t **indirection_a, const int input_channel,
                              const int kernel_size, int8_t *packed_ptr, int32_t *sums_ptr);

    void pack_8bit_neon_weight(const int8_t *w, const int input_channel,
                               const int kernel_size, int8_t *packed_ptr, int32_t *sums_ptr,
                               const int8_t input_zero_point, const int8_t kernel_zero_point);
}
#endif //QUANT_CONV_PACK_KERNEL_H
