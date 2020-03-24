//
// Created by pfzhang on 20/3/2020.
//

#include <gtest/gtest.h>
#include "convolution-operator-tester.h"

TEST(CONVOLUTION_OP, 3X3) {

    ConvolutionOperatorTester()
    .inputSize(56,56)
    .padding(1)
    .kernelSize(3)
    .inputChannels(32)
    .outputChannels(64)
    .iterations(3)
    .activationBits(1)
    .weightBits(1)
    .test();
}