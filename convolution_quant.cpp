//
// Created by pfzhang on 20/3/2020.
//
// C header
#include <stddef.h>
#include <stdint.h>
// C++ header
#include <iostream>
#include <atomic>

#include "convolution.h"
#include "math.h"
#include "params.h"
#include "qconv_kernel.h"
#include "pack.h"
#include "block_map.h"

namespace quant_conv {

#if 0 
    // Here, we just show the reference code.
    bool quant_conv_run_conv_reference(conv_operator_t op) {
        const size_t batch_size = op->batch_size;
        const size_t input_channels = op->input_channels;
        const size_t output_channels = op->output_channels;
        const uint32_t mr = 4;
        const uint32_t nr = 4;
        const uint32_t kr = 16;
        const size_t k_stride = (input_channels + (kr - 1)) & -kr;
        const size_t n_stride = (output_channels + (nr - 1)) & -nr;
        
        const size_t output_size = op->output_height * op->output_width;
        const size_t kernel_size = op->kernel_height * op->kernel_width;
        //const size_t m_stride = (output_size + (mr - 1)) & -mr
        const size_t m_stride = round_up(output_size, mr);

        const int8_t** indirection_a = (const int8_t**) op->indirection_buffer;
        const int8_t* packed_w = (const int8_t*) op->packed_weight;
        int32_t* output = (int32_t*) op->output;

        // The layout of the weight: Co-HWCi-NrKr
        // The layout of the indirection buffer: batch_size –> tile_outer -> kernel_size-> tile_offset
        // The layout of the input is HWCi
        // The layout of the output is HWCo
        for(size_t image = 0; image < batch_size; image ++) {
            for(size_t pixel = 0; pixel < output_size; pixel += mr) {
                const size_t mr_block_size = min(output_size - pixel, mr);
                for(size_t mi = 0; mi < mr_block_size; mi++){
                    for(size_t oc_start = 0; oc_start < output_channels; oc_start += nr) {
                        const size_t nr_block_size = min(output_channels - oc_start, nr);
                        for(size_t k_off = 0; k_off < kernel_size; k_off ++){

                            size_t buffer_index = image * m_stride * kernel_size + pixel * kernel_size + k_off * mr + mi;
                            const int8_t* input_pointer = indirection_a[buffer_index];

                            for(size_t ic_start = 0; ic_start < input_channels; ic_start += kr){
                                const size_t kr_block_size = min(input_channels - ic_start, kr);

                                for (size_t ni = 0; ni < nr_block_size; ni++) {

                                    size_t out_idx = image * output_size * output_channels + (pixel + mi) * output_channels +
                                            oc_start + ni;
                                    for (size_t ki = 0; ki < kr_block_size; ki++) {
                                        size_t w_idx =
                                                oc_start * (kernel_size * k_stride + sizeof(int32_t)) + nr * sizeof(int32_t)
                                                + k_off * k_stride * nr + ic_start * nr + ni * kr + ki;

                                         output[out_idx] += (int32_t)input_pointer[ic_start + ki] * (int32_t)packed_w[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;

    }


    bool quant_conv_run_conv(conv_operator_t op){

       // std::cout << "Enter Here" << std::endl;
        const size_t batch_size = op->batch_size;
        const size_t input_channels = op->input_channels;
        const size_t output_channels = op->output_channels;
        const uint32_t mr = 4;
        const uint32_t nr = 4;
        const uint32_t kr = 16;

        const size_t k_stride = (input_channels + (kr - 1)) & -kr;
        const size_t n_stride = (output_channels + (nr - 1)) & -nr;

        const size_t output_size = op->output_height * op->output_width;
        const size_t kernel_size = op->kernel_height * op->kernel_width;
        //const size_t m_stride = (output_size + (mr - 1)) & -mr
        const size_t m_stride = round_up(output_size, mr);

        const int8_t** indirection_a = (const int8_t**) op->indirection_buffer;
        const void* packed_w = (void*) op->packed_weight;
        int32_t* output = (int32_t*) op->output;

        const qconv_neon_params neon_params = (qconv_neon_params){
            .kernel_zero_point = op->kernel_zero_point
        };


        for(size_t image = 0; image < batch_size; image++){
            for(size_t mr_block_start = 0; mr_block_start < output_size; mr_block_start += mr) {

                const void * pw = packed_w;
                size_t mr_block_size = min(output_size - mr_block_start, mr);
                for(size_t nr_block_start = 0; nr_block_start < output_channels; nr_block_start += nr) {
                    size_t nr_block_size = min(output_channels - nr_block_start, nr);

                    compute_quant_kernel_a7w7(mr_block_size, nr_block_size, input_channels, kernel_size,
                            indirection_a, // The start position of indirection buffer
                            pw,  // The start position of packed weights
                            output + nr_block_start, // The start position of output.
                            output_channels,
                            neon_params  // Here, transfer the kernel zero point to the ukernel.
                            );

                    pw = (void*) ((uintptr_t) pw + nr * (k_stride * kernel_size + sizeof(int32_t)));
                }
                // After computing mr * output_channels output pixels
                indirection_a += mr * kernel_size;
                // Here, we must add mr_block_size, it can avoid overwrite something wrongly.
                output += mr_block_size * output_channels;
            }
        }
        return true;
    }

#endif

    bool quant_conv_run_conv_with_packed_input(conv_operator_t op)
    {

        const BlockMap& block_map = *(op->block_map);

        const qconv_neon_params neon_params = (qconv_neon_params){
                .kernel_zero_point = op->kernel_zero_point
        };
        const int nb = num_blocks(block_map); // nb: num_blocks
        //bool* local_packed = (bool*) malloc(nb * sizeof(bool));

        Pair<int> block;
        Pair<int> start;
        Pair<int> end;

        const int8_t** indirection_a = (const int8_t**) op->indirection_buffer;
        int8_t* packed_input = op->packed_input;
        int32_t* input_sums = op->input_sums;
        int8_t* packed_weight = (int8_t*) op->packed_weight;
        int32_t* weight_sums = op->weight_sums;
        int kernel_size = op->kernel_height * op->kernel_width;
        int input_channels = op->input_channels;
        int output_channels = op->output_channels;

        int32_t* output = (int32_t*) op->output;
        int block_id = 0;

        int depth = kernel_size * input_channels;

        std::cout << "The number of blocks is " << nb << std::endl;
        while(block_id < nb){

            get_block_by_index(block_map, block_id, &block);
            get_block_matrix_coords(block_map, block, &start, &end);

            pack_input(block_map, indirection_a, packed_input, input_sums,
                    start[Side::kLhs], end[Side::kLhs], kernel_size, input_channels);



            compute_quant_kernel_with_packed_input_a7w7(
                    packed_input + start[Side::kLhs] * depth, // The start position of packed_input
                    input_sums + start[Side::kLhs], // The start position of input_sums
                    packed_weight + start[Side::kRhs] * depth, // The start position of packed weight
                    weight_sums + start[Side::kRhs], // The start position of weight sums.
                    output + start[Side::kLhs] * output_channels + start[Side::kRhs], // The start position of output
                    input_channels, kernel_size, output_channels,
                    start[Side::kLhs], end[Side::kLhs],
                    start[Side::kRhs], end[Side::kRhs],
                    neon_params
            );



            block_id += 1;

        }


    }

} // namespace quant_conv