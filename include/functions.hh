#ifndef MACHINE_LEARNING_FUNCTIONS_HH
#define MACHINE_LEARNING_FUNCTIONS_HH

namespace Functions {
    template<typename T>
    float sigmoid(T x) {
        return 1 / (1 + std::exp(-x));
    }

    template<typename T>
    float sigmoid_derivative(T x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }
}

#endif //MACHINE_LEARNING_FUNCTIONS_HH
