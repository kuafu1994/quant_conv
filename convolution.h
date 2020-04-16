
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdint.h>
#include <stddef.h>

#include "block_map.h"

namespace quant_conv {

typedef struct conv_operator {
    size_t batch_size;
    uint32_t padding_top;
    uint32_t padding_bottom;
    uint32_t padding_left;
    uint32_t padding_right;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_height;
    uint32_t stride_width;
    size_t input_channels;
    size_t output_channels;

    size_t input_height;
    size_t input_width;
    const void * input;
    //size_t input_pixel_stride;
    
    int8_t kernel_zero_point;

    int kernel_rows;
    int kernel_cols;

    size_t output_height;
    size_t output_width;
    void* output;
    //size_t output_pixel_stride;

    int8_t* packed_weight;
    
    //void* zero_buffer;
    void* zero_pointer;
    //void* lookup_table;

    const void** indirection_buffer;

    int32_t* input_sums;
    int8_t* packed_input;

    int32_t* weight_sums;

    BlockMap* block_map;

    int thread_count;

    int activation_bits;
    int weight_bits;

} conv_operator;

typedef conv_operator* conv_operator_t;

    bool quant_conv2d_create_pipeline(
            uint32_t padded_top, uint32_t padded_bottom, uint32_t padded_left, uint32_t padded_right,
            uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,
            size_t input_channels, size_t output_channels,
            int8_t input_zero_point,
            int8_t kernel_zero_point,
            conv_operator_t* convolution_out
    );


    bool quant_conv2d_setup_nhwc(conv_operator_t convolution,
                                 size_t batch_size, size_t input_height, size_t input_width,
                                 const int8_t* input,
                                 const int8_t* kernel,
                                 int32_t* output,
                                 const int8_t input_zero_point);



bool quant_conv_run_conv(conv_operator_t op);
bool quant_conv_run_conv_reference(conv_operator_t op);
bool quant_conv_run_conv_with_packed_input(conv_operator_t op);
bool qconv_delete(conv_operator_t convolution);

bool quant_conv_run_conv_with_packed_input_no_block_map(conv_operator_t op);

} // namespace quant_conv
#endif 