//
// Created by pfzhang on 20/3/2020.
//

#include "detect_arm.h"

#if defined _linux__ && defined __aarch64__
#include <sys/auxv.h>
#endif

namespace quant_conv {

namespace {
#if defined __linux__ && defined __aarch64__

        bool DetectDotProdByLinuxAuxvMethod(){
            // This is the value of HWCAP_ASIMDDP in sufficiently recent Linux headers,
            // however we need to support building against older headers for the time
            // being.
            const int kLocalHwcapAsimddp = 1 << 20;
            return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
        }
#endif
} // namespace

bool DetectDotprod() {

#if defined __linux__ && defined __aarch64__
    return DetectDotprodByLinuxAuxvMethod();
#endif

    return false;
}

}