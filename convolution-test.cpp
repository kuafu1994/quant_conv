//
// Created by pfzhang on 20/3/2020.
//

#include <gtest/gtest.h>
#include "convolution-operator-tester.h"


TEST(VGG, CONV2) {

    ConvolutionOperatorTester()
    .inputSize(224,224)
    .padding(1)
    .kernelSize(3)
    .inputChannels(64)
    .outputChannels(64)
    .iterations(1)
    .activationBits(8)
    .weightBits(8)
    .test();
}

#if 0

TEST(VGG, CONV3) {

    ConvolutionOperatorTester()
            .inputSize(112,112)
            .padding(1)
            .kernelSize(3)
            .inputChannels(64)
            .outputChannels(128)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

TEST(VGG, CONV4) {

    ConvolutionOperatorTester()
            .inputSize(112,112)
            .padding(1)
            .kernelSize(3)
            .inputChannels(128)
            .outputChannels(128)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

TEST(VGG, CONV5) {

    ConvolutionOperatorTester()
            .inputSize(56,56)
            .padding(1)
            .kernelSize(3)
            .inputChannels(128)
            .outputChannels(256)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}


TEST(VGG, CONV6) {

    ConvolutionOperatorTester()
            .inputSize(56,56)
            .padding(1)
            .kernelSize(3)
            .inputChannels(256)
            .outputChannels(256)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}



TEST(VGG, CONV7) {

    ConvolutionOperatorTester()
            .inputSize(56,56)
            .padding(1)
            .kernelSize(3)
            .inputChannels(256)
            .outputChannels(256)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

TEST(VGG, CONV8) {

    ConvolutionOperatorTester()
            .inputSize(28,28)
            .padding(1)
            .kernelSize(3)
            .inputChannels(256)
            .outputChannels(512)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

TEST(VGG, CONV9) {

    ConvolutionOperatorTester()
            .inputSize(28,28)
            .padding(1)
            .kernelSize(3)
            .inputChannels(512)
            .outputChannels(512)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

TEST(VGG, CONV10) {

    ConvolutionOperatorTester()
            .inputSize(28,28)
            .padding(1)
            .kernelSize(3)
            .inputChannels(512)
            .outputChannels(512)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}


TEST(VGG, CONV11) {

    ConvolutionOperatorTester()
            .inputSize(14,14)
            .padding(1)
            .kernelSize(3)
            .inputChannels(512)
            .outputChannels(512)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

TEST(VGG, CONV12) {

    ConvolutionOperatorTester()
            .inputSize(14,14)
            .padding(1)
            .kernelSize(3)
            .inputChannels(512)
            .outputChannels(512)
            .iterations(3)
            .activationBits(8)
            .weightBits(8)
            .test();
}

#endif