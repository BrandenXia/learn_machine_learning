#include <cmath>

#include "functions.hh"

namespace Functions {
    template<typename T>
    requires std::is_arithmetic_v<T>
    float sigmoid(T x) {
        return 1 / (1 + std::exp(-x));
    }

    template<typename T>
    requires std::is_arithmetic_v<T>
    float sigmoid_derivative(T x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }
}