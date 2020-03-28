//
// Created by PENGFEI ZHANG on 2020/3/26.
//

#ifndef QUANT_CONV_COMMON_H
#define QUANT_CONV_COMMON_H

#ifndef QCONV_INTERNAL
#if defined(__ELF__)
#define QCONV_INTERNAL __attribute__((__visibility__("internal")))
#elif defined(__MACH__)
#define QNNP_INTERNAL __attribute__((__visibility__("hidden")))
  #else
    #define QNNP_INTERNAL
#endif
#endif
#endif //QUANT_CONV_COMMON_H
