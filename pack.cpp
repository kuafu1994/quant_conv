//
// Created by PENGFEI ZHANG on 2020/4/2.
//

#include "pack.h"

namespace quant_conv {
    void pack_weight(const BlockMap& block_map, const int8_t* weight, int8_t* packed_weight, int32_t* weight_sums,
                     const int8_t input_zero_point, const int8_t kernel_zero_point,
                     const int start, const int end, const int kernel_size, const int input_channels){

        assert(start % 4 == 0);
        assert(end % 4 == 0);
        const int stride = kernel_size * input_channels;

        for(int idx = start; idx < end; idx+=4){

            const int8_t* w = weight + idx * stride;

            int8_t* packed_ptr = packed_weight + idx * stride;
            int32_t* sums_ptr = weight_sums + idx;

            pack_8bit_neon_weight(w, input_channels, kernel_size, packed_ptr, sums_ptr,
                                  input_zero_point, kernel_zero_point);
        }

    }

    // Here, stride = kh * kw * ic;
    void pack_input(const BlockMap& block_map, const int8_t** indirection_a, int8_t* packed_input, int32_t* input_sums,
                    const int start, const int end, const int kernel_size, const int input_channel) {

        assert(start % 4 == 0);
        const int stride = kernel_size * input_channel;
        for(int idx = start; idx < end; idx += 4){
            const int8_t** a = indirection_a + kernel_size * idx; // The indirection buffer.

            int8_t* packed_ptr = packed_input + stride * idx;
            int32_t* sums_ptr = input_sums + idx;
            pack_8bit_neon_input(a, input_channel, kernel_size, packed_ptr, sums_ptr);
        }
    }
}