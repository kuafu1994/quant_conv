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

        uint32_t activation_max = (1 << activation_bits_) - 1;
        uint32_t weight_max = (1 << weight_bits_) - 1;
        auto activation_rng = std::bind(std::uniform_int_distribution<uint8_t>(0, activation_max), rng);
        auto weight_rng = std::bind(std::uniform_int_distribution<uint8_t>(0, weight_max), rng);

        // Here, we assume that the storage layout of input is HWC.
        std::vector<uint8_t> input(batch_size_ * input_width_ * input_height_ * input_channels_);
        // Here, we assume that the kernel layout is OHWI.
        std::vector<uint8_t> kernel(output_channels_ * kernel_height_ * kernel_width_ * input_channels_);
        std::vector<uint8_t> output(batch_size_ * outputHeight() * outputWidth() * output_channels_);

        // Only in the worst case, we will use int32_t to accumulate the values.
        std::vector<uint32_t> accumulators(batch_size_ * outputHeight() * outputWidth() * output_channels_);

        std::vector<uint32_t> results(batch_size_ * outputHeight() * outputWidth() * output_channels_);

        // Here, we set input_zero_point as activation_max/2, for example, if the number of activation bits is 4,
        // then the activation_max is 2^4-1=15 and the input_zero_point is 7.
        const uint8_t input_zero_point = activation_max / 2;

        // It is the same as input_zero_point.
        const uint8_t kernel_zero_point = weight_max / 2;

        for(size_t iteration = 0; iteration < iterations(); iteration++){

            std::generate(input.begin(), input.end(), std::ref(activation_rng));
            std::generate(kernel.begin(), kernel.end(), std::ref(weight_rng));
            std::fill(output.begin(), output.end(), 0xA5);
            std::fill(accumulators.begin(), accumulators.end(), 0);

            for(size_t i = 0; i < batch_size_; i++){
                for(size_t oy = 0; oy < outputHeight(); oy++){
                    for(size_t ox = 0; ox < outputWidth(); ox ++){
                        for(size_t ky = 0; ky < kernel_height_; ky++){

                            const size_t iy = oy * stride_height_ - padding_top_;

                            if(iy < input_height_){
                                for(size_t kx = 0; kx < kernel_width_; kx++){
                                    const size_t ix = ox * stride_width_ - padding_left_;

                                    if(ix < input_width_){
                                        for(size_t oc = 0; oc < output_channels_; oc++){
                                            for(size_t ic = 0; ic < input_channels_; ic++){
                                                accumulators[(((i * outputHeight() + oy) * outputWidth()) + ox) * output_channels_ + oc] +=
                                                        uint32_t(input[(((i * input_height_ + iy) * input_width_) + ix) * input_channels_ + ic]) *
                                                        uint32_t(kernel[((oc * kernel_height_ + ky) * kernel_width_) * input_channels_ + ic]);
                                            } // for ic
                                        } // for oc
                                    } // if ix
                                } // for kx
                            } // if iy
                        } // for ky
                    } // for ox
                } // for oy
            } // for i

            conv_operator_t convolution;
           // The convolution code here.

           // create the conv2d pipeline here.
           ASSERT_EQ(true, quant_conv::quant_conv2d_create_pipeline(
                padding_top_, padding_bottom_, padding_left_, padding_right,
                kernel_height_, kernel_width_, stride_height_, stride_width_, 
                input_channels_, output_channels_, 
                input_zero_point, kernel_zero_point,
                kernel.data(), &convolution
           ));

           // setup the conv2d.
           
           size_t input_pixel_stride = input_channels_;
           size_t output_pixel_stride = output_channels_;
           ASSERT_EQ(true, quant_conv::quant_conv2d_setup_nhwc(
               batch_size_, input_height_, input_width_, input.data(),
               input_pixel_stride, results.data(), output_pixel_stride
           ));

           ASSERT_EQ(true, quant_conv_run_conv2d(convolution));
        

            for(size_t i = 0; i < batch_size_; i ++){
                for(size_t y = 0; y < outputHeight(); y++) {
                    for(size_t x = 0; x < outputWidth(); x++){
                        for(size_t c = 0; c < output_channels_; c++){
                            ASSERT_EQ(accumulators[((i * outputHeight() + y) * outputWidth() + x) * output_channels_ + c], results[((i * outputHeight() + y) * outputWidth() + x) * output_channels_ + c]);
                        }
                    }
                }
            }
        }
    }

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
