
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

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
    size_t input_pixel_stride;
    
    uint8_t kernel_zero_point;

    size_t output_height;
    size_t output_width;
    void* output;
    size_t output_pixel_stride;

    void* packed_weight;
    
    void* zero_buffer;
    void* zero_pointer;
    void* lookup_table;

    const void** indirection_buffer;
} conv_operator;

typedef conv_operator* conv_operator_t;


bool quant_conv2d_create_pipeline(
    uint32_t padded_top, uint32_t padded_bottom, uint32_t padded_left, uint32_t padded_right, 
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,
    size_t input_channels, size_t output_channels, 
    uint8_t input_zero_point,
    uint8_t kernel_zero_point, 
    const uint8_t* kernel, conv_operator_t* convolution_out
);

bool quant_conv2d_setup_nhwc(conv_operator_t convolution,
        size_t batch_size, size_t input_height, size_t input_width, 
        const uint8_t* input, size_t input_pixel_stride, 
        uint8_t* output, size_t output_pixel_stride);
} // namespace quant_conv
#endif 