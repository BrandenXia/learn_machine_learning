#ifndef MACHINE_LEARNING_STATS_HH
#define MACHINE_LEARNING_STATS_HH

#include <map>
#include <type_traits>
#include <utility>
#include <ranges>
#include <numeric>

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
    auto mean(T &data) {
        return std::accumulate(data.begin(), data.end(), 0.0f) /
               static_cast<float>(distance(data.begin(), data.end()));
    }

    template<ArithmeticIterable T>
    auto variance(T &data) {
        const auto m = mean(data);
        return accumulate(data.begin(), data.end(), 0.0f, [m](const auto &acc, const auto &x) {
            return acc + (x - m) * (x - m);
        }) / static_cast<float>(distance(data.begin(), data.end()));
    }

    template<Arithmetic T>
    auto covariance(const std::map<T, T> &data) {
        const auto v_x = data | std::ranges::views::keys;
        const auto v_y = data | std::ranges::views::values;
        const auto x_mean = mean(v_x);
        const auto y_mean = mean(v_y);
        return accumulate(data.begin(), data.end(), 0.0f, [x_mean, y_mean](const auto &acc, const auto &pair) {
            return acc + (pair.first - x_mean) * (pair.second - y_mean);
        }) / static_cast<float>(distance(data.begin(), data.end()));
    }

    template<Arithmetic T>
    std::pair<float, float> linear_regression(const std::map<T, T> &data) {
        const auto v_x = data | std::ranges::views::keys;
        const auto v_y = data | std::ranges::views::values;

        const auto x_mean = mean(v_x);
        const auto y_mean = mean(v_y);
        const auto cov = covariance(data);
        const auto a = cov / variance(v_x);
        const auto b = y_mean - a * x_mean;
        return {a, b};
    };

    template<typename T, typename V>
    float mse_loss(T &data, V &predictions) {
        std::vector<float> diff{};
        diff.resize(std::distance(data.begin(), data.end()));
        std::transform(data.begin(), data.end(), predictions.begin(), diff.begin(), std::minus{});
        std::transform(diff.begin(), diff.end(), diff.begin(), diff.begin(), std::multiplies{});
        return mean(diff);
    }
}

#endif //MACHINE_LEARNING_STATS_HH
