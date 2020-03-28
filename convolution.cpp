
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "indirection.h"
#include "convolution.h"
#include "pack.h"
#include "params.h"

namespace quant_conv {

static inline size_t compute_output_dimension(
    size_t padded_input_dimension, 
    size_t kernel_dimension,
    size_t stride_dimension
) {
    return (padded_input_dimension - kernel_dimension) / stride_dimension + 1;
}


// TOOD: should template it.
bool quant_conv2d_create_pipeline(
    uint32_t padded_top, uint32_t padded_bottom, uint32_t padded_left, uint32_t padded_right, 
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,
    size_t input_channels, size_t output_channels, 
    int8_t input_zero_point,
    int8_t kernel_zero_point,
    const int8_t* kernel, conv_operator_t* convolution_out
){
    conv_operator_t convolution = NULL;
    // malloc the memory for convolution.
    convolution = (conv_operator_t)calloc(1, sizeof(conv_operator));
    
    if(convolution == NULL){
        std::cout << "failed to allocate " << sizeof(struct conv_operator) << " bytes for conv_operator structure" << std::endl;
        return false;
    }

    size_t kernel_size = kernel_height * kernel_width;

    // nr and kr are set emprically.
    const uint32_t nr = 4;
    const uint32_t kr = 16;
    
    // here kr and nr must be power of 2.
    const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
    const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;
    
    // Now, prepare the packed weights.
    const size_t packed_weights_size = (sizeof(int8_t) * kernel_size * k_stride + sizeof(int32_t)) * n_stride;
    // packed_weight is of type void*
    convolution->packed_weight = malloc(packed_weights_size);
    
    if(convolution->packed_weight == NULL){
        std::cout << "fail to allocate " << packed_weights_size << " bytes for packed weights" << std::endl;
        return false;
    }
    // packed_weight is initialized withe kernel_zero_point.
    memset(convolution->packed_weight, kernel_zero_point, packed_weights_size);
    pack_weight(output_channels, kernel_size, input_channels,
                nr, kr, input_zero_point, kernel_zero_point,
                kernel, (void*)((uintptr_t) convolution->packed_weight));

    // Now, prepare the padding zeros
    const bool any_padding = (padded_top | padded_bottom | padded_left | padded_right); 
    
    size_t zero_size = sizeof(int8_t) * k_stride;

    if(any_padding){
        void* zero_pointer = malloc(zero_size);
        if(zero_pointer == NULL){
            std::cout << "failed to allocate " << zero_size << " bytes for zero padding" << std::endl;
            return false;
        }
        memset(zero_pointer, input_zero_point, zero_size);
        //memset(zero_pointer, 0, zero_size);
        convolution->zero_pointer = zero_pointer;
    }

    convolution->padding_top = padded_top;
    convolution->padding_bottom = padded_bottom;
    convolution->padding_left = padded_left;
    convolution->padding_right = padded_right;

    convolution->kernel_height = kernel_height;
    convolution->kernel_width = kernel_width;
    convolution->stride_height = stride_height;
    convolution->stride_width = stride_width;
    convolution->input_channels = input_channels;
    convolution->output_channels = output_channels;
    convolution->kernel_zero_point = kernel_zero_point;
    *convolution_out = convolution;
    return true;
}

// TODO: template it;
bool quant_conv2d_setup_nhwc(conv_operator_t convolution,
        size_t batch_size, size_t input_height, size_t input_width, 
        const int8_t* input,
        int32_t* output)
    {

        convolution->batch_size = batch_size;
        convolution->input_height = input_height;
        convolution->input_width = input_width;
        convolution->input = input;
        // Should we still keep the input_pixel_stride?
        //convolution->input_pixel_stride = input_pixel_stride;
        
        convolution->output_height = compute_output_dimension(
            convolution->padding_top + input_height + convolution->padding_bottom,
            convolution->kernel_height, convolution->stride_height
        );
        
        convolution->output_width = compute_output_dimension(
            convolution->padding_left + input_width + convolution->padding_right,
            convolution->kernel_width, convolution->stride_width
        );

        convolution->output = output;

        //convolution->output_pixel_stride = output_pixel_stride;

        const size_t kernel_height = convolution->kernel_height;
        const size_t kernel_width = convolution->kernel_width;
        const size_t kernel_size = kernel_height * kernel_width;
        const size_t output_height = convolution->output_height;
        const size_t output_width = convolution->output_width;
        const size_t output_size = output_height * output_width;

        const size_t output_tile_size = 4;  // mr

        const size_t tiled_output_size = round_up(output_size, output_tile_size);
        
        const size_t indirection_buffer_size = sizeof(void*) * batch_size * tiled_output_size * kernel_size;

        const void** indirection_buffer = (const void**) realloc(convolution->indirection_buffer, indirection_buffer_size);

        if(indirection_buffer == NULL){
            std::cout << "failed to allocate " << indirection_buffer_size << " bytes for indirection buffer" << std::endl;
            return false;
        }

        convolution->indirection_buffer = indirection_buffer;

        quant_indirection_init_conv2d(convolution, output_tile_size, tiled_output_size);    

        return true;            
    }
} // namespace quant_conv

