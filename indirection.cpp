
#include <stddef.h>
#include "math.h"

namespace quant_conv {

    void quant_indirection_init_conv2d(
        conv_operator_t op,
        size_t output_tile_size,
        size_t tiled_output_size
    ){
        const void** indirection_buffer   = op->indirection_buffer;
        const void* input                 = op->input;
        const size_t input_pixel_stride   = op->input_pixel_stride;
        const void* zero                  = op->zero_pointer;
        const size_t input_channels       = op->input_channels;
        const size_t batch_size           = op->batch_size;
        const size_t input_height         = op->input_height;
        const size_t input_width          = op->input_width;
        const size_t output_height        = op->output_height;
        const size_t output_width         = op->output_width;
        const size_t kernel_height        = op->kernel_height;
        const size_t kernel_width         = op->kernel_width;
        const size_t stride_height        = op->stride_height;
        const size_t stride_width         = op->stride_width;
        const size_t input_padding_top    = op->padding_top;
        const size_t input_padding_left   = op->padding_left;

        const size_t output_size = output_height * output_width;
        const size_t kernel_size = kernel_height * kernel_width;

        for(size_t image = 0; image < batch_size; image++){
            
            for(size_t tile_start = 0; tile_start < tiled_output_size; tile_start += output_tile_size){
                
                for(size_t tile_offset = 0; tile_offset < output_tile_size; tile_offset ++){
                    const size_t toi = tile_start + tile_offset;
                    const size_t oi = min(toi, output_size - 1);

                    const size_t output_y = oi / output_width;
                    const size_t output_x = oi % output_width;

                    
                    for(size_t kernel_y = 0; kernel_y < kernel_height; kernel_y ++){
                        
                        const size_t input_y = output_y * stride_height - input_padding_top;
                        
                        if(input_y < input_height) {
                            
                            for(size_t kernel_x = 0; kernel_x < kernel_width; kernel_x ++){
                                
                                const size_t input_x = output_x * stride_width - input_padding_left;
                                //const size_t index = image * tiled_output_size * kernel_size + tile_start * kernel_size + (kernel_y * kernel_width + kernel_x) * output_tile_size + output_tile_offset;

                                // batch_size –> tile_outer -> kernel_size-> tile_offset
                                const size_t index = image * tiled_output_size * kernel_size + tile_start * kernel_size + (kernel_y * kernel_width + kernel_x) * output_tile_size + tile_offset;
                                if(input_x < input_width){
                                    // The input layout is NHWC.
                                    indirection_buffer[index] = (char*)input + ((image * input_height + input_y) * input_width + input_x) * input_pixel_stride;
                                } else {
                                    indirection_buffer[index] = zero;
                                }
                            }
                        } else {
                            for(size_t kernel_x = 0; kernel_x < kernel_width; kernel_x ++){
                                const size_t index = image * tiled_output_size * kernel_size + tile_start * kernel_size + (kernel_y * kernel_width + kernel_x) * output_tile_size + tile_offset;
                                indirection_buffer[index] = zero;
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace quant_conv