
#ifndef INDIRECTION_H
#define INDIRECTION_H


namespace quant_conv {
    
     void quant_indirection_init_conv2d(
        conv_operator_t op,
        size_t output_tile_size,
        size_t tiled_output_size
    );
    

} // namespace quant_conv
#endif