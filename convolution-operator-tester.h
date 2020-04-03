//
// Created by pfzhang on 20/3/2020.
//

#ifndef QUANT_CONV_CONVOLUTION_OPERATOR_TESTER_H
#define QUANT_CONV_CONVOLUTION_OPERATOR_TESTER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <stdint.h>
#include "convolution.h"
#include <iostream>
class ConvolutionOperatorTester {
public:
    inline ConvolutionOperatorTester& padding(uint32_t padding){
        this->padding_top_ = padding;
        this->padding_bottom_ = padding;
        this->padding_left_ = padding;
        this->padding_right_ = padding;

        return *this;
    }

    inline uint32_t paddingTop() const{
        return padding_top_;
    }

    inline uint32_t paddingBottom() const{
        return padding_bottom_;
    }

    inline uint32_t paddingLeft() const{
        return padding_left_;
    }

    inline uint32_t paddingRight() const{
        return padding_right_;
    }

    inline ConvolutionOperatorTester& inputSize(uint32_t input_height, uint32_t input_width){

        assert(input_height >= 1);
        assert(input_width >= 1);

        this->input_height_ = input_height;
        this->input_width_ = input_width;
        return *this;
    }

    inline uint32_t inputHeight() const {
        return input_height_;
    }

    inline uint32_t inputWidth() const {
        return input_width_;
    }

    inline ConvolutionOperatorTester& inputChannels(size_t input_channels){

        assert(input_channels >= 1);
        this->input_channels_ = input_channels;

        return *this;
    }

    inline ConvolutionOperatorTester& outputChannels(size_t output_channels){

        assert(output_channels >= 1);
        this->output_channels_ = output_channels;

        return *this;
    }

    inline size_t inputChannels() const {
        return input_channels_;
    }

    inline size_t outputChannels() const {
        return output_channels_;
    }

    inline ConvolutionOperatorTester& batchSize(size_t batch_size) {

        assert(batch_size >= 1);
        this->batch_size_ = batch_size;
        return *this;
    }

    inline size_t batchSize() const {

        return this->batch_size_;
    }

    inline ConvolutionOperatorTester& kernelSize(uint32_t kernel_size){
        assert(kernel_size >= 1);

        this->kernel_height_ = kernel_size;
        this->kernel_width_ = kernel_size;

        return *this;
    }

    inline uint32_t kernelHeight() const {

        return kernel_height_;
    }

    inline uint32_t kernelWidth() const {

        return kernel_width_;
    }

    inline ConvolutionOperatorTester& stride(uint32_t stride){

        assert(stride >= 1);
        this->stride_height_ = stride;
        this->stride_width_ = stride;

        return *this;
    }

    inline ConvolutionOperatorTester& stride(uint32_t stride_height, uint32_t stride_width){

        assert(stride_height >= 1);
        assert(stride_width >= 1);
        this->stride_height_ = stride_height;
        this->stride_width_ = stride_width;

        return *this;
    }

    inline uint32_t strideHeight() const {
        return stride_height_;
    }

    inline uint32_t strideWidth() const {
        return stride_width_;
    }

    inline size_t outputHeight() const {

        const size_t padding_input_height = padding_top_ + input_height_ + padding_bottom_;

        return (padding_input_height - kernel_height_) / stride_height_ + 1;
    }

    inline size_t outputWidth() const {

        const size_t padding_input_width = padding_left_ + input_width_ + padding_right_;

        return (padding_input_width - kernel_width_) / stride_width_ + 1;
    }

    inline ConvolutionOperatorTester& activationBits(uint32_t value){
        assert(value >= 1);
        this->activation_bits_ = value;
        return *this;
    };

    inline ConvolutionOperatorTester& weightBits(uint32_t value) {
        assert(value >= 1);
        this->weight_bits_ = value;
        return *this;
    }

    inline uint32_t activationBits() const {
        return activation_bits_;
    }

    inline uint32_t weightBits() const {
        return weight_bits_;
    }

    inline ConvolutionOperatorTester& iterations(size_t value){
        assert(value >= 1);
        this->iterations_ = value;

        return *this;
    }

    inline size_t iterations() const {
        return this->iterations_;
    }

    void test() const {

        std::random_device rd;
        auto rng = std::mt19937(rd());

        int32_t activation_max = (1 << activation_bits_ - 1) - 1;
        int32_t activation_min = -activation_max;
        int32_t weight_max = (1 << weight_bits_ - 1) - 1;
        int32_t weight_min = -weight_max;
        auto activation_rng = std::bind(std::uniform_int_distribution<int8_t>( activation_min, activation_max), rng);
        auto weight_rng = std::bind(std::uniform_int_distribution<int8_t>(weight_min, weight_max), rng);

        // Here, we assume that the storage layout of input is HWC.
        std::vector<int8_t> input(batch_size_ * input_width_ * input_height_ * input_channels_);
        // Here, we assume that the kernel layout is OHWI.
        std::vector<int8_t> kernel(output_channels_ * kernel_height_ * kernel_width_ * input_channels_);

        std::vector<int8_t> output(batch_size_ * outputHeight() * outputWidth() * output_channels_);

        // Only in the worst case, we will use int32_t to accumulate the values.
        std::vector<int32_t> results_ref(batch_size_ * outputHeight() * outputWidth() * output_channels_);

        // We will store the results in these vector.
        std::vector<int32_t> results(batch_size_ * outputHeight() * outputWidth() * output_channels_);

        int8_t input_zero_point = 1;

        int8_t kernel_zero_point = 1;

        for(size_t iteration = 0; iteration < iterations(); iteration++){

            std::generate(input.begin(), input.end(), std::ref(activation_rng));
            std::generate(kernel.begin(), kernel.end(), std::ref(weight_rng));
            //std::fill(input.begin(), input.end(), 2);
            //std::fill(kernel.begin(), kernel.end(), 2);
            std::fill(results_ref.begin(), results_ref.end(), 0);
            std::fill(results.begin(), results.end(), 0);

            quant_conv::conv_operator_t convolution;
            // The convolution code here.

            // create the conv2d pipeline here.
            quant_conv::quant_conv2d_create_pipeline(
                    padding_top_, padding_bottom_, padding_left_, padding_right_,
                    kernel_height_, kernel_width_, stride_height_, stride_width_,
                    input_channels_, output_channels_,
                    input_zero_point, kernel_zero_point,
                    &convolution
            );

            // setup the conv2d.
            size_t input_pixel_stride = input_channels_;
            size_t output_pixel_stride = output_channels_;

            quant_conv::quant_conv2d_setup_nhwc(convolution,
                                                batch_size_, input_height_, input_width_, input.data(), kernel.data(),
                                                results.data(), input_zero_point);

            quant_conv::quant_conv_run_conv_with_packed_input(convolution);

            for(size_t i = 0; i < batch_size_; i++){
                for(size_t oy = 0; oy < outputHeight(); oy++){
                    for(size_t ox = 0; ox < outputWidth(); ox++){
                        for(size_t ky = 0; ky < kernel_height_; ky++){
                            const size_t iy = oy * stride_height_ + ky - padding_top_;
                            // size_t -> uint64_t, so if iy is negative, then iy must be larger than input_height_;
                            if(iy < input_height_){
                                for(size_t kx = 0; kx < kernel_width_; kx++){
                                    const size_t ix = ox * stride_width_ + kx - padding_left_;
                                    if(ix < input_width_){
                                        for(size_t oc = 0; oc < output_channels_; oc++){
                                            for(size_t ic = 0; ic < input_channels_; ic++){
                                                results_ref[(((i * outputHeight() + oy) * outputWidth()) + ox) * output_channels_ + oc] +=
                                                        (int32_t(input[(((i * input_height_ + iy) * input_width_) + ix) * input_channels_ + ic]) - int32_t(input_zero_point)) *
                                                        (int32_t(kernel[((oc * kernel_height_ + ky) * kernel_width_ + kx) * input_channels_ + ic]) - int32_t(kernel_zero_point));
                                            } // for ic
                                        } // for oc
                                    } // if ix
                                } // for kx
                            } // if iy
                        } // for ky
                    } // for ox
                } // for oy
            } // for i


            for(size_t i = 0; i < batch_size_; i ++){
                for(size_t y = 0; y < outputHeight(); y++) {
                    for(size_t x = 0; x < outputWidth(); x++){
                        for(size_t c = 0; c < output_channels_; c++){
                       //for(size_t c = 0; c < 4; c++) {
                            const size_t index = ((i * outputHeight() + y) * outputWidth() + x) * output_channels_ + c;
                            if(results_ref[index] != results[index]) {
                               std::cout << results_ref[index] << "," << results[index] << ":" << results_ref[index] - results[index] << ":" << x <<", " << y << "," << c << std::endl;
                            }
                            ASSERT_EQ(results_ref[index], results[index])
                            << "y=" << y << "," << "x=" << x << "," << "c=" << c << "\n";
                        }
                    }
                }
            }

        } // for iterations.
    } // test()

private:
    uint32_t padding_top_{0};
    uint32_t padding_bottom_{0};
    uint32_t padding_left_{0};
    uint32_t padding_right_{0};

    size_t input_height_{1};
    size_t input_width_{1};
    size_t input_channels_{1};
    size_t output_channels_{1};

    size_t batch_size_{1};
    size_t kernel_height_{1};
    size_t kernel_width_{1};

    size_t stride_height_{1};
    size_t stride_width_{1};

    uint32_t activation_bits_{1};
    uint32_t weight_bits_{1};

    size_t iterations_{1};
};



#endif //QUANT_CONV_CONVOLUTION_OPERATOR_TESTER_H
