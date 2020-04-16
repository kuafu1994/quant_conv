//
// Created by pfzhang on 19/3/2020.
//

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "convolution.h"


static void convolution_a1w1(benchmark::State& state, const char* net){
    const size_t batch_size = state.range(0);
    const size_t input_height = state.range(1);
    const size_t input_width = state.range(2);
    const size_t kernel_height = state.range(3);
    const size_t kernel_width = state.range(4);
    const size_t stride = state.range(5);
    const size_t input_channels = state.range(6);
    const size_t output_channels = state.range(7);

    // rd must be callable
    std::random_device rd;
    auto rng = std::mt19937(rd());

    // u8rng will produce binary value between 0 and 1.
    auto u8rng = std::bind(std::uniform_int_distribution<int8_t>(-2, 1), rng);

    // define the padding
    const size_t padding_left = kernel_width / 2;
    const size_t padding_right = kernel_width - 1 - padding_left;

    const size_t padding_top = kernel_height / 2;
    const size_t padding_bottom = kernel_height - 1 - padding_top;

    const size_t output_height = (padding_top + input_height + padding_bottom - kernel_height) / stride + 1;
    const size_t output_width = (padding_left + input_width + padding_right - kernel_width) / stride + 1;

    // generate the input.
    std::vector<int8_t> input(batch_size * input_height * input_width * input_channels);
    std::generate(input.begin(), input.end(), std::ref(u8rng));

    // generate the kernel, the input should be OHWI
    std::vector<int8_t> kernel(output_channels * kernel_height * kernel_width * input_channels);
    std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));

    // generate the bias,
    std::vector<int32_t> output(batch_size * output_height * output_width * output_channels);

    int8_t input_zero_point = 1;
    int8_t kernel_zero_point = 1;

    quant_conv::conv_operator_t convolution;

    bool pipeline_create = quant_conv::quant_conv2d_create_pipeline(
            padding_top, padding_bottom, padding_left, padding_right,
            kernel_height, kernel_width, stride, stride,
            input_channels, output_channels,
            input_zero_point, kernel_zero_point,
            &convolution);

    if(!pipeline_create){
        state.SkipWithError("failed to create convolution pipeline!");
    }

    bool setup = quant_conv::quant_conv2d_setup_nhwc(convolution,
                                        batch_size, input_height, input_width,
                                        input.data(),
                                        kernel.data(),
                                        output.data(),
                                        input_zero_point);

    if(!setup){
        state.SkipWithError("failed to setup Convolution operator");
    }


    for(auto _ : state){
        auto start = std::chrono::high_resolution_clock::now();

        quant_conv::quant_conv_run_conv_with_packed_input(convolution);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

      //  std::cout << "time: " << elapsed_seconds.count() * 1e3 << std::endl;
        state.SetIterationTime(elapsed_seconds.count());
    }

    bool del = quant_conv::qconv_delete(convolution);

    if(!del){
        state.SkipWithError("failed to delete the operator");
    }

    state.SetItemsProcessed(
            uint64_t(state.iterations()) * 2 *
            batch_size * output_height * output_width
             * input_channels * output_channels *
            kernel_height * kernel_width);
}


static void ResNet18(benchmark::internal::Benchmark* b) {
    b->ArgNames({"N", "H", "W", "KH", "KW", "S", "GCin", "GCout"});

    /********************* Conv 1 *********************/
    /*       N   H    W   KH  KW  S  Cin  GCout */
    //b->Args({1, 224, 224,  7,  7, 2, 3,  64});
    /******************** Conv 2.X ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  56,  56,  3,  3, 1, 64,  64});
    /******************** Conv 3.X ********************/
    /*       N   H    W   KH  KW  S Cin  GCout */
    b->Args({1,  56,  56,  3,  3, 2, 64,  128});
    b->Args({1,  28,  28,  3,  3, 1, 128, 128});
    b->Args({1,  56,  56,  1,  1, 2, 64,  128});
    /******************** Conv 4.X ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  28,  28,  3,  3, 2, 128,  256});
    b->Args({1,  14,  14,  3,  3, 1, 256,  256});
    b->Args({1,  28,  28,  1,  1, 2, 128,  256});
    /******************** Conv 5.X ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  16,  16,  3,  3, 2, 256,  512});
    b->Args({1,   8,   8,  3,  3, 1, 512,  512});
    b->Args({1,  16,  16,  1,  1, 2, 256,  512});
}


static void ResNet50(benchmark::internal::Benchmark* b) {
    b->ArgNames({"N", "H", "W", "KH", "KW", "S", "Cin", "Cout"});

    /********************* Conv 1 *********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1, 224, 224,  7,  7,  2,  3,  64});
    /******************** Conv 2.1 ********************/

    b->Args({1,  56,  56,  1,  1, 1, 64, 64});
    b->Args({1,  56,  56,  3,  3, 1, 64, 64});
    b->Args({1,  56,  56,  1,  1, 1, 64, 256});
/*b->Args({1,  56,  56,  1,  1, 1, 1, 1,   64,  256});*/
    /******************** Conv 2.X ********************/

    b->Args({1,  56,  56,  1,  1, 1, 256, 64});
/*b->Args({1,  56,  56,  3,  3, 1, 1, 1,   64,   64});*/
/*b->Args({1,  56,  56,  1,  1, 1, 1, 1,   64,  256});*/
    /******************** Conv 3.1 ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  56,  56,  1,  1, 1, 256,  128});
    b->Args({1,  56,  56,  3,  3, 2, 128,  128});
    b->Args({1,  28,  28,  1,  1, 1, 128,  512});
    b->Args({1,  56,  56,  1,  1, 2, 256,  512});
    /******************** Conv 3.X ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  28,  28,  1,  1, 1, 512, 128});
    b->Args({1,  28,  28,  3,  3, 1, 128,  128});
/*b->Args({1,  28,  28,  1,  1, 1, 1, 1,  128,  512});*/
    /******************** Conv 4.1 ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  28,  28,  1,  1, 1,  512, 256});
    b->Args({1,  28,  28,  3,  3, 2,  256, 256});
    b->Args({1,  14,  14,  1,  1, 1,  256, 1024});
    b->Args({1,  28,  28,  1,  1, 2,  512, 1024});
    /******************** Conv 4.X ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  14,  14,  1,  1, 1, 1024,  256});
    b->Args({1,  14,  14,  3,  3, 1, 256,  256});
/*b->Args({1,  14,  14,  1,  1, 1, 1, 1,  256, 1024});*/
    /******************** Conv 5.1 ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,  14,  14,  1,  1, 1, 1024,  512});
    b->Args({1,  14,  14,  3,  3, 2, 512,  512});
    b->Args({1,   7,   7,  1,  1, 1, 512, 2048});
    b->Args({1,  14,  14,  1,  1, 2, 1024, 2048});
    /******************** Conv 5.X ********************/
    /*       N   H    W   KH  KW  S  Cin  Cout */
    b->Args({1,   7,   7,  1,  1, 1, 2048,  512});
    b->Args({1,   7,   7,  3,  3, 1, 512,  512});

}


static void VGG(benchmark::internal::Benchmark* b) {
    b->ArgNames({"N", "H", "W", "KH", "KW", "S", "Cin", "Cout"});

    /********************* Conv 1.1 ********************/
    /*b, h, w, kh, kw, s, ci, co*/
   // b->Args({1, 224, 224,  3,  3, 1, 3, 64});
    /********************* Conv 1.2 ********************/
    /*b, h, w, kh, kw, s, ci, co*/
    b->Args({1, 224, 224,  3,  3, 1, 64, 64});

    b->Args({1, 224, 224, 3, 3, 1, 64, 128});

    /********************* Conv 2.1 ********************/
    /*b, h, w, kh, kw, s, ci, co*/
    b->Args({1, 112, 112,  3,  3, 1, 64, 128});
    /********************* Conv 2.2 ********************/
    /*b, h, w, kh, kw, s, ci, co*/
    b->Args({1, 112, 112,  3,  3, 1, 128, 256});

    /********************* Conv 3.1 ********************/
    /*b, h, w, kh, kw, s, ci, co*/
    b->Args({1,  56,  56,  3,  3, 1, 256, 256});
    /********************* Conv 3.2 ********************/
    /*b, h, w, kh, kw, s, ci, co*/
    b->Args({1,  56,  56,  3,  3, 1, 256, 256});
    /********************* Conv 3.3 ********************/

    b->Args({1,  56,  56,  3,  3, 1, 256, 512});

    /********************* Conv 4.1 ********************/

    b->Args({1,  28,  28,  3,  3, 1, 512, 512});
    /********************* Conv 4.2 ********************/

    b->Args({1,  28,  28,  3,  3, 1, 512, 512});
    /********************* Conv 4.3 ********************/

    b->Args({1,  28,  28,  3,  3, 1, 512, 512});

    b->Args({1, 28, 28, 3, 3, 1, 512, 512});



    /********************* Conv 5.X ********************/

    b->Args({1,  14,  14,  3,  3, 1, 512,  512});
    /********************* Conv 5.3 ********************/

    b->Args({1,  14,  14,  3,  3, 1, 512,  512});

    b->Args({1, 14, 14, 3, 3, 1, 512, 512});
    b->Args({1, 14, 14, 3, 3, 1, 512, 512});
}

BENCHMARK_CAPTURE(convolution_a1w1, resnet18, "VGG")->Apply(ResNet18);

BENCHMARK_MAIN();




