//
// Created by PENGFEI ZHANG on 2020/3/31.
//

#include "block_map.h"
#include "math.h"
#include <vector>
#include <assert.h>
#include <limits>
#include <iostream>
#include <stdint.h>

namespace quant_conv {

    void get_block_by_index(const BlockMap& block_map, int index, Pair<int>* block){

        const uint32_t index_u32 = index;
        const uint32_t num_blocks_per_local_curve = 1u << (2 * block_map.num_blocks_base_log2);

        const uint32_t n1 = index_u32 & (num_blocks_per_local_curve - 1);

        Pair<int> local_pos;

        if(block_map.traversal_order == BlockMapTraversalOrder::kLinear){

            local_pos[Side::kLhs] = n1 & ((1u << block_map.num_blocks_base_log2) - 1);
            local_pos[Side::kRhs] = n1 >> block_map.num_blocks_base_log2;

        } else {
            // Decode fractal z-order
            const uint32_t n2 = (n1 & 0x99999999u) | ((n1 & 0x44444444u) >> 1) |
                                     ((n1 & 0x22222222u) << 1);
            const uint32_t n4 = (n2 & 0xc3c3c3c3u) | ((n2 & 0x30303030u) >> 2) |
                                     ((n2 & 0x0c0c0c0cu) << 2);
            const uint32_t n8 = (n4 & 0xf00ff00fu) | ((n4 & 0x0f000f00u) >> 4) |
                                     ((n4 & 0x00f000f0u) << 4);
            const uint32_t n16 = (n8 & 0xff0000ffu) | ((n8 & 0x00ff0000u) >> 8) |
                                      ((n8 & 0x0000ff00u) << 8);
            local_pos[Side::kLhs] = n16 & 0xffff;
            local_pos[Side::kRhs] = n16 >> 16;
            if (block_map.traversal_order == BlockMapTraversalOrder::kFractalU) {
                // Change fractal z-order to u-order
                local_pos[Side::kLhs] ^= local_pos[Side::kRhs];
            }
        }

        const uint32_t rectangular_index = index_u32 >> (2 * block_map.num_blocks_base_log2);

        for(Side side : {Side::kLhs, Side::kRhs}) {
            const uint32_t mask = (1u << block_map.rectangularness_log2[side]) - 1;
            const int rectangular_offset = (rectangular_index & mask) << block_map.num_blocks_base_log2;

            (*block)[side] = rectangular_offset + local_pos[side];
        }
    }


    int get_cache_locality_score(int block_size_log2, int rows, int cols, int depth, int kernel_rows_log2, int kernel_cols_log2) {


        if(rows <= (1 << kernel_rows_log2) || cols <= (1 << kernel_cols_log2)){
            return 0;
        }

        const int block_rows = std::min(1 << block_size_log2, rows);
        const int block_cols = std::min(1 << block_size_log2, cols);

        const int kLocalDataCacheSizeLog2 = 15;

        const int lhs_bytes_log2 = pot_log2(1) + ceil_log2(block_rows * depth);
        const int rhs_bytes_log2 = pot_log2(1) + ceil_log2(block_cols * depth);

        const int total_bytes_log2 = 1 + std::max(lhs_bytes_log2, rhs_bytes_log2);
        const int nonlocality_log2 = total_bytes_log2 - kLocalDataCacheSizeLog2;

        if(nonlocality_log2 < -1){
            return 64;
        } else if(nonlocality_log2 == -1){
            return 56;
        } else if(nonlocality_log2 == 0) {
            return 48;
        } else if(nonlocality_log2 == 1){
            return 32;
        } else if(nonlocality_log2 == 2){
            return 0;
        } else {
            return -64;
        }
    }


    void get_block_matrx_coords(Side side, const BlockMap& block_map, const int block, int* start, int* end){

        // Every blocks will be appended with a miss kernel.
        *start = block * block_map.small_block_dims[side] + std::min(block, block_map.large_blocks[side])
                                                            * block_map.kernel_dims[side];
        *end = *start + block_map.small_block_dims[side] +
                (block < block_map.large_blocks[side]? block_map.kernel_dims[side]: 0);

        assert(*start % block_map.kernel_dims[side] == 0);
        assert(*end % block_map.kernel_dims[side] == 0);
        assert(*end <= block_map.dims[side]);
        assert(*start < *end);
        assert(*start >= 0);
    }


    void get_block_matrix_coords(const BlockMap& block_map, const Pair<int>& block, Pair<int>*start, Pair<int> *end){

        for(Side side: {Side::kLhs, Side::kRhs}){
            get_block_matrx_coords(side, block_map, block[side], &(*start)[side], &(*end)[side]);
        }

    }


    void GetRectangularness(int rows, int cols, int kernel_rows, int kernel_cols,
                            int *rows_rectangularness_log2,
                            int *cols_rectangularness_log2) {

        *rows_rectangularness_log2 = 0;
        *cols_rectangularness_log2 = 0;

        const int min_kernel_inner_loop_runs_log2 = 3;
        if (rows > cols) {
            // In the col dimension, the log2 of loop runs.
            int cols_of_kernel_inner_loop_runs_log2 = ceil_log2(cols) - pot_log2(kernel_cols);
            int min_rows_of_kernel_inner_loop_runs_log2 =
                    std::max(0, min_kernel_inner_loop_runs_log2 - cols_of_kernel_inner_loop_runs_log2);
            // 0, floor_log2(rows) - pot_log2(kernel_rows) - min_rows_of_kernel_inner_loop_runs_log2, floor_log2_quotient.
            *rows_rectangularness_log2 =
                    std::min(floor_log2_quotient(rows, cols), std::max(0, floor_log2(rows) - pot_log2(kernel_rows) - min_rows_of_kernel_inner_loop_runs_log2));
            // Sanity check that we did not over-estimate rows_rectangularness_log2.
            //RUY_DCHECK_GE(rows >> *rows_rectangularness_log2, cols);
            assert((rows >> *rows_rectangularness_log2) >= cols);
        } else if (cols > rows) {
            int rows_of_kernel_inner_loop_runs_log2 = ceil_log2(rows) - pot_log2(kernel_rows);
            int min_cols_of_kernel_inner_loop_runs_log2 = std::max(0, min_kernel_inner_loop_runs_log2 - rows_of_kernel_inner_loop_runs_log2);
            *cols_rectangularness_log2 = std::min(floor_log2_quotient(cols, rows), std::max(0, floor_log2(cols) - pot_log2(kernel_cols) - min_cols_of_kernel_inner_loop_runs_log2));
            // Sanity check that we did not over-estimate cols_rectangularness_log2.
            //  RUY_DCHECK_GE(cols >> *cols_rectangularness_log2, rows);
            assert((cols >> *cols_rectangularness_log2 >= rows));
        }
        assert(!*rows_rectangularness_log2 || !*cols_rectangularness_log2);
    }

// Computes a 'multithreading score'. When multithreading, we need there to
// be at least as many tiles as there are threads, and hopefully
// substantially more than that, so we benefit from ruy's ability to
// dispatch fine-grained workloads to threads.
    int GetMultithreadingScore(int block_size_log2, int rows, int cols,
                               int tentative_thread_count) {
        const int num_full_blocks_of_rows = rows >> block_size_log2;
        const int num_full_blocks_of_cols = cols >> block_size_log2;
        const int candidate_num_full_blocks_log2 = floor_log2(
                std::max(1, num_full_blocks_of_rows * num_full_blocks_of_cols));

        // The values here have been tuned on ARM Cortex-A55.
        // We expect this to have to be tuned differently for other CPUs.
        if (tentative_thread_count == 1) {
            return 0;
        } else {
            const int blocks_per_thread_log2 = candidate_num_full_blocks_log2 - ceil_log2(tentative_thread_count);
            if (blocks_per_thread_log2 < 0) {
                return -64;
            } else if (blocks_per_thread_log2 == 0) {
                return -16;
            } else if (blocks_per_thread_log2 == 1) {
                return -8;
            } else if (blocks_per_thread_log2 == 2) {
                return 0;
            } else if (blocks_per_thread_log2 == 3) {
                return 8;
            } else {
                return 16;
            }
        }
    }

    // Compute a 'kernel amortization score'. This is the notion that very small
// tiles result in more overhead outside of kernels, more complex memory
// access patterns and less benefits from ruy's fat kernels, so we reward
// larger blocks more than smaller ones.
    int GetKernelAmortizationScore(int block_size_log2, int rows, int cols,
                                   int kernel_rows_log2, int kernel_cols_log2) {
        const int block_rows = std::min(1 << block_size_log2, rows);
        const int block_cols = std::min(1 << block_size_log2, cols);
        const int kernels_per_block_log2 =
                floor_log2(block_rows * block_cols) - kernel_rows_log2 - kernel_cols_log2;
        // RUY_DCHECK_GE(kernels_per_block_log2, 0);
        assert(kernels_per_block_log2 >= 0);
        // The values here have been tuned on ARM Cortex-A55.
        // We expect this to have to be tuned differently for other CPUs.
        if (kernels_per_block_log2 == 0) {
            return 0;
        } else if (kernels_per_block_log2 == 1) {
            return 8;
        } else if (kernels_per_block_log2 == 2) {
            return 16;
        } else if (kernels_per_block_log2 == 3) {
            return 24;
        } else if (kernels_per_block_log2 == 4) {
            return 32;
        } else if (kernels_per_block_log2 == 5) {
            return 40;
        } else if (kernels_per_block_log2 == 6) {
            return 48;
        } else if (kernels_per_block_log2 == 7) {
            return 56;
        } else {
            return 64;
        }
    }


    // rows: the number of rows in the output matrix;
    // cols: the number of cols in the output matrix;
    // The kernel_rows is 4 and kernel_cols is 4 in most cases.
    void make_block_map(const int rows, const int cols, const int depth, const int kernel_rows,
                        const int kernel_cols, const int tentative_thread_count,
                        BlockMap *block_map) {

        // We define it as kFractalU now.

        // TODO: study other approachblock_map->traversal_order = BlockMapTraversalOrder::kFractalU;
        block_map->traversal_order = BlockMapTraversalOrder::kLinear;

        int rows_rectangularness_log2 = 0;
        int cols_rectangularness_log2 = 0;

        GetRectangularness(rows, cols, kernel_rows, kernel_cols, &rows_rectangularness_log2,
                           &cols_rectangularness_log2);

        const int kernel_rows_log2 = pot_log2(kernel_rows);
        const int kernel_cols_log2 = pot_log2(kernel_cols);
        const int kernel_size_log2 = std::max(kernel_rows_log2, kernel_cols_log2);

        const int size = std::min(rows, cols); // size is 64.
        const int size_log2 = std::max(kernel_size_log2, floor_log2(size)); //

        assert(size_log2 >= kernel_size_log2);

        // Log 2 of maximum kernel each block.
        static constexpr int kMaxKernelsPerBlockLog2 = 6;

        // The best block size should be tuned here.
        // TODO: automatically tune the best block_size_log2
        const int max_block_size_log2 = std::min(size_log2, kernel_size_log2 + kMaxKernelsPerBlockLog2);
        int best_score = std::numeric_limits<int>::min();
        int best_score_block_size_log2 = -1;
        for (int block_size_log2 = kernel_size_log2; block_size_log2 <= max_block_size_log2; block_size_log2++) {
            const int multithreading_score = GetMultithreadingScore(block_size_log2, rows, cols,
                                                                    tentative_thread_count);
            const int cache_locality_score = get_cache_locality_score(
                      block_size_log2, rows, cols, depth, kernel_rows_log2, kernel_cols_log2);
            const int kernel_amortization_score = GetKernelAmortizationScore(block_size_log2, rows, cols,
                                                                             kernel_rows_log2, kernel_cols_log2);
           const int score = multithreading_score + kernel_amortization_score;
            //const int score = cache_locality_score + kernel_amortization_score;

            if (score > best_score) {
                best_score = score;
                best_score_block_size_log2 = block_size_log2;
            }
        }

        int num_blocks_base_log2 = size_log2 - best_score_block_size_log2;

      //
      // std::cout << "The num blocks base log2 " << num_blocks_base_log2 << std::endl;
       // num_blocks_base_log2 = 0;
        assert(num_blocks_base_log2 >= 0);

        const int num_blocks_of_rows_log2 =
                num_blocks_base_log2 + rows_rectangularness_log2;
        const int num_blocks_of_cols_log2 =
                num_blocks_base_log2 + cols_rectangularness_log2;

        const int smallr = round_down_pot(rows >> num_blocks_of_rows_log2, kernel_rows);
        const int smallc = round_down_pot(cols >> num_blocks_of_cols_log2, kernel_cols);

        const int missr = round_up_pot(rows - (smallr << num_blocks_of_rows_log2), kernel_rows)
                >> pot_log2(kernel_rows);
        const int missc = round_up_pot(cols - (smallc << num_blocks_of_cols_log2), kernel_cols)
                >> pot_log2(kernel_cols);

        block_map->dims[Side::kLhs] = rows;
        block_map->dims[Side::kRhs] = cols;
        block_map->kernel_dims[Side::kLhs] = kernel_rows;
        block_map->kernel_dims[Side::kRhs] = kernel_cols;

        block_map->num_blocks_base_log2 = num_blocks_base_log2;
        block_map->rectangularness_log2[Side::kLhs] = rows_rectangularness_log2;
        block_map->rectangularness_log2[Side::kRhs] = cols_rectangularness_log2;

        block_map->small_block_dims[Side::kLhs] = smallr;
        block_map->small_block_dims[Side::kRhs] = smallc;

        block_map->large_blocks[Side::kLhs] = missr;
        block_map->large_blocks[Side::kRhs] = missc;

        block_map->thread_count = std::min(tentative_thread_count, num_blocks(*block_map));
    }
} // namespace quant_conv.