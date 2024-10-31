#ifndef MACHINE_LEARNING_STATS_HH
#define MACHINE_LEARNING_STATS_HH

#include <map>
#include <type_traits>
#include <utility>

namespace Stats {
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T>;

    template<typename T>
    concept ArithmeticIterable = requires(T t) {
        t.begin();
        t.end();
        std::is_arithmetic_v<decltype(*t.begin())>;
    };

    template<ArithmeticIterable T>
    auto mean(T &data);

    template<ArithmeticIterable T>
    auto variance(T &data);

    template<Arithmetic T>
    auto covariance(const std::map<T, T> &data);

    template<Arithmetic T>
    std::pair<float, float> linear_regression(const std::map<T, T> &data);

    template<ArithmeticIterable T, ArithmeticIterable V>
    auto mse_loss(T &data, V &predictions);
}

#endif //MACHINE_LEARNING_STATS_HH
