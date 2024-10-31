#ifndef MACHINE_LEARNING_FUNCTIONS_HH
#define MACHINE_LEARNING_FUNCTIONS_HH

#include <type_traits>

namespace Functions {
    template<typename T>
    requires std::is_arithmetic_v<T>
    float sigmoid(T x);

    template<typename T>
    requires std::is_arithmetic_v<T>
    float sigmoid_derivative(T x);
}

#endif //MACHINE_LEARNING_FUNCTIONS_HH
