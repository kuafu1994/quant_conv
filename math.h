
#ifndef MATH_H
#define MATH_H

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <vector>
namespace quant_conv {



    inline size_t min(size_t a, size_t b){
        return a < b? a : b;
    }

    inline size_t divide_round_up(size_t n, size_t q){
        return n % q == 0? n/q : n/q + 1;
    }

    inline size_t round_up(size_t n, size_t q){
        return divide_round_up(n, q) * q;
    }

    template<typename Integer>
    Integer floor_log2(Integer n){
        static_assert(std::is_integral<Integer>::value, "");
        static_assert(std::is_signed<Integer>::value, "");
        static_assert(sizeof(Integer) == 4 || sizeof(Integer) == 8, "");

        assert(n >= 1);

        if (sizeof(Integer) == 4) {
            return 31 - __builtin_clz(n);
        } else {
            return 63 - __builtin_clzll(n);
        }

    }
    template<typename Integer>
    Integer ceil_log2(Integer n){
        assert(n >= 1);
        return n == 1 ? 0 : floor_log2(n-1) + 1;
    }

    template <typename Integer>
    bool is_pot(Integer value) {
        return (value > 0) && ((value & (value - 1)) == 0);
    }

    template <typename Integer>
    Integer pot_log2(Integer n) {
       // RUY_DCHECK(is_pot(n));
        assert(is_pot(n));
        return floor_log2(n);
    }

    template <typename Integer, typename Modulo>
    Integer round_down_pot(Integer value, Modulo modulo) {
        //RUY_DCHECK_EQ(modulo & (modulo - 1), 0);
        assert((modulo & (modulo - 1)) == 0);
        return value & ~(modulo - 1);
    }

    template <typename Integer, typename Modulo>
    Integer round_up_pot(Integer value, Modulo modulo) {
        return round_down_pot(value + modulo - 1, modulo);
    }

    // The floor log2 of quotient of num and denom.
    template <typename Integer>
    Integer floor_log2_quotient(Integer num, Integer denom) {
        if (num <= denom) {
            return 0;
        }

        int log2_quotient = floor_log2(num) - ceil_log2(denom);
        if ((denom << (log2_quotient + 1)) <= num) {
            log2_quotient++;
        }
        return log2_quotient;
    }
} // namespace quant_conv

#endif 