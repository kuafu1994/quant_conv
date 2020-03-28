
#ifndef MATH_H
#define MATH_H

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
} // namespace quant_conv

#endif 