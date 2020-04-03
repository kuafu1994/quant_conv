//
// Created by PENGFEI ZHANG on 2020/3/31.
//

#ifndef QUANT_CONV_BLOCK_MAP_H
#define QUANT_CONV_BLOCK_MAP_H

#include "pair.h"

namespace quant_conv {
    enum class BlockMapTraversalOrder {
        // Plain old row-by-row or column-by-column traversal.
                kLinear,
        // Fractal Z-order curve,
                kFractalZ,
        // variant of Z-order doing a U instead of Z.
                kFractalU
    };

    struct BlockMap {

        // the number of threads to be used
        int thread_count;

        // The order in order to traversal the matrix of which this BlockMap represents
        // a tiling
        BlockMapTraversalOrder traversal_order;

        // The dimensions of the block_map, that is, of the destination
        // matrix rounded up to next multiples of kernel_dims
        Pair<int> dims;

        int num_blocks_base_log2;

        Pair<int> rectangularness_log2;

        Pair<int> kernel_dims;

        Pair<int> small_block_dims;

        Pair<int> large_blocks;


    };

    void get_block_by_index(const BlockMap& block_map, int index, Pair<int>* block);

    void get_block_matrix_coords(const BlockMap& block_map, const Pair<int>& block, Pair<int>*start, Pair<int> *end);

    void make_block_map(const int rows, const int cols, const int kernel_rows,
                        const int kernel_cols, const int tentative_thread_count, BlockMap *block_map) ;

    inline int num_blocks(const BlockMap &block_map) {
        return 1 << (2 * block_map.num_blocks_base_log2 + block_map.rectangularness_log2[Side::kLhs] +
                     block_map.rectangularness_log2[Side::kRhs]);
    }

    inline int num_block_per_side(Side side, const BlockMap& block_map){

        return 1 << (block_map.num_blocks_base_log2 + block_map.rectangularness_log2[side]);
    }

}

#endif //QUANT_CONV_BLOCK_MAP_H
