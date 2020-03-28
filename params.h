//
// Created by PENGFEI ZHANG on 2020/3/26.
//

#ifndef QUANT_CONV_PARAMS_H
#define QUANT_CONV_PARAMS_H

#include <stdint.h>

namespace quant_conv {

    typedef struct qconv_parameters {
        uint8_t mr;
        uint8_t nr;
        uint8_t kr;
    } qconv_parameters;


    typedef struct qconv_neon_params {
        int8_t kernel_zero_point;
        int8_t input_zero_point;
    } qconv_neon_params;
}
#endif //QUANT_CONV_PARAMS_H
