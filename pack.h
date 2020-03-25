
#ifndef PACK_H
#define PACK_H

#include "math.h"

namespace quant_conv {

// The original weight is CoHWCi,
// after the packing, the layout is Co-HWCi-NrKr
static inline void pack_weight(size_t output_channels, size_t ks, size_t input_channels, 
                                uint32_t nr, uint32_t kr, 
                                uint8_t izp, uint8_t kzp, 
                                const uint8_t* k, void* packed_w){
     
     // equations (5) (6) of https://arxiv.org/pdf/1806.08342.pdf
     // kernel_size * input_channel * izp * kzp

     const int32_t off = (int32_t) ks * (int32_t) input_channels * (int32_t) izp * (int32_t) kzp;
     
     for(size_t nr_block_start = 0; nr_block_start < output_channels; nr_block_start += nr){
         
         int32_t *packed_off = (int32_t*) packed_w;

         const size_t nr_block_size = min(output_channels - nr_block_start, nr);
         
         for(size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset ++){
             *((int32_t*) packed_w) = off;
             packed_w = (void*) ((uintptr_t) packed_w + sizeof(int32_t));
         }
         
         packed_w = (void*) ((uintptr_t) packed_w + (nr - nr_block_size) * sizeof(int32_t));

         for(size_t ki = 0; ki < ks; ki++){
             for(size_t kr_block_start = 0; kr_block_start < input_channels; kr_block_start += kr){
                 const size_t kr_block_size = min(input_channels - kr_block_start, kr);
                 
                 for(size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset ++){
                     uint32_t ksum = 0;
                     for(size_t kr_block_offset = 0; kr_block_offset < kr_block_size; kr_block_offset++){
                         const uint8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * input_channels + kr_block_start + kr_block_offset];
                         ksum += (uint32_t) kv;
                         *((uint8_t*) packed_w) = kv;
                         packed_w = (void*)((uintptr_t) packed_w + sizeof(uint8_t));
                     }
                     packed_off[nr_block_offset] -= ksum * izp;
                     packed_w = (void*) ((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(uint8_t));
                 }
                 packed_w = (void*) ((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(uint8_t));
             } 
         }   
     }    
}

} // namespace quant_conv

#endif 
