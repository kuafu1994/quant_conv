//
// Created by PENGFEI ZHANG on 2020/3/31.
//

#ifndef QUANT_CONV_PAIR_H
#define QUANT_CONV_PAIR_H
#include <assert.h>
namespace quant_conv {

    enum class Side { kLhs = 0, kRhs = 1};
    template <typename T>
    class Pair final {
    public:
        Pair() {}
        Pair(const T& a, const T& b): elem_{a, b}{
        }

        const T& operator[](Side side) const {
            const int index = static_cast<int>(side);
            return elem_[index];
        }

        T& operator[] (Side side){
            const int index = static_cast<int>(side);
            assert(index == 0 || index == 1);

            return elem_[index];
        }

    private:
        static_assert(static_cast<int>(Side::kLhs) == 0, "");
        static_assert(static_cast<int>(Side::kRhs) == 1, "");
        T elem_[2];
    };
}
#endif //QUANT_CONV_PAIR_H
