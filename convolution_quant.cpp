//
// Created by pfzhang on 20/3/2020.
//

#include "convolution.h"
namespace quant_conv {

    // Here, we just show the reference code.
    bool quant_conv_run_conv(conv_operator_t op) {
        const size_t batch_size = op->batch_size;
        const size_t input_channels = op->input_channels;
        const size_t output_channels = op->output_channels;
        const uint32_t mr = 8;
        const uint32_t nr = 8;
        const uint32_t kr = 8;
        const size_t k_stride = (input_channels + (kr - 1)) & -kr;
        const size_t n_stride = (output_channels + (nr - 1)) & -nr;
        
        const size_t output_size = op->output_height * op->output_width;
        const size_t kernel_size = op->kernel_height * op->kernel_width;
        const size_t m_stride = (output_size + (mr - 1)) & -mr;

        // The layout of the weight: Co-HWCi-NrKr
        // The layout of the indirection buffer: batch_size –> tile_outer -> kernel_size-> tile_offset
        for(size_t image = 0; image < batch_size; image ++) {
            
            
            
            
        }
    
        

        


        return true;

    }
    
} // namespace quant_conv