#include <iostream>
#include <cmath>
#include <type_traits>
#include <array>
#include <numeric>
#include <random>

#include "functions.hh"
#include "stats.hh"

using std::array, std::inner_product, std::function, std::cout;

namespace NeuralNetwork {
    class Activation {
    public:
        virtual float operator()(float x) = 0;

        virtual float derivative(float x) = 0;

        virtual ~Activation() = default;
    };

    class Sigmoid : public Activation {
    public:
        constexpr float operator()(float x) final { return Functions::sigmoid(x); }

        constexpr float derivative(float x) final { return Functions::sigmoid_derivative(x); }
    };

    template<size_t Input, size_t Output>
    class Layer {
    public:
        std::shared_ptr<Activation> activation;
        array<array<float, Input>, Output> weights;
        array<float, Output> biases;

        explicit Layer(std::shared_ptr<Activation> acf = std::make_shared<Sigmoid>())
                : activation(std::move(acf)) {
            std::random_device rd;
            std::mt19937 gen{rd()};
            std::uniform_real_distribution<float> dist(0, 1);

            // xavier initialization for weights
            float xavier = std::sqrt(6.0f / (Input + Output));
            for (auto &w: weights) {
                for (auto &w_i: w) {
                    w_i = dist(gen) * 2 * xavier - xavier;
                }
            }

            // random initialization for biases
            for (auto &b: biases) { b = dist(gen); }
        }

        array<float, Output> sum(const array<float, Input> &inputs) {
            array<float, Output> outputs{};

            std::transform(
                    weights.begin(), weights.end(), biases.begin(), outputs.begin(),
                    [inputs](const auto &w, float &b) {
                        return inner_product(inputs.begin(), inputs.end(), w.begin(), b);
                    }
            );

            return outputs;
        }

        array<float, Output> activate(const array<float, Output> &sums) {
            array<float, Output> outputs{};
            std::transform(sums.begin(), sums.end(), outputs.begin(), [this](const auto &s) {
                return (*activation)(s);
            });
            return outputs;
        }

        array<float, Output> operator()(const array<float, Input> &inputs) {
            return activate(sum(inputs));
        }
    };

    template<size_t Input, size_t Hidden, size_t Output> requires (Input > 0, Hidden > 0)
    class NeuralNetwork {
    public:
        NeuralNetwork() : hidden_layer(), output_layer() {}

        template<size_t Size>
        void train(const array<array<float, Input>, Size> &data, const array<array<float, Output>, Size> &labels) {
            for (int i = 0; i < Size; ++i) {
                auto sum_h = hidden_layer.sum(data[i]);
                auto h = hidden_layer.activate(sum_h);

                auto sum_o = output_layer.sum(h);
                auto o = output_layer.activate(sum_o);

                // backpropagation
                array<float, Output> output_error{};
                std::transform(o.begin(), o.end(), labels[i].begin(), output_error.begin(), std::minus{});
            }
        }

    private:
        Layer<Input, Hidden> hidden_layer;
        Layer<Hidden, Output> output_layer;
    };
}

int main() {
    NeuralNetwork::NeuralNetwork<2, 2, 1> nn;

    array<array<float, 2>, 4> data = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};

    array<array<float, 1>, 4> labels = {{{0}, {1}, {1}, {0}}};

    nn.train(data, labels);
}
