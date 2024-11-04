#include <iostream>
#include <cmath>
#include <type_traits>
#include <array>
#include <numeric>
#include <random>
#include <algorithm>
#include <ranges>

#include "functions.hh"
#include "stats.hh"
#include "linear_algebra.hh"


using std::array, std::inner_product, std::function, std::cout;
using LinearAlgebra::Vector;

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
        array<Vector<Input>, Output> weights;
        Vector<Output> biases;

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

        Vector<Output> sum(const Vector<Input> &inputs) {
            Vector<Output> outputs{};

            std::transform(
                    weights.begin(), weights.end(), biases.begin(), outputs.begin(),
                    [inputs](const auto &w, float &b) { return w.dot(inputs) + b; }
            );

            return outputs;
        }

        Vector<Output> activate(const Vector<Output> &sums) {
            Vector<Output> outputs{};
            std::transform(sums.begin(), sums.end(), outputs.begin(), [this](const auto &s) {
                return (*activation)(s);
            });
            return outputs;
        }

        Vector<Output> operator()(const Vector<Input> &inputs) {
            return activate(sum(inputs));
        }
    };

    template<size_t Input, size_t Hidden, size_t Output> requires (Input > 0, Hidden > 0)
    class NeuralNetwork {
    public:
        NeuralNetwork() : hidden_layer(), output_layer() {}

        Vector<Output> operator()(const Vector<Input> &inputs) {
            return output_layer(hidden_layer(inputs));
        }

        template<size_t Size>
        void epoch(const array<Vector<Input>, Size> &data, const array<Vector<Output>, Size> &labels) {
            for (int i = 0; i < Size; ++i) {
                auto sum_h = hidden_layer.sum(data[i]);
                auto h = hidden_layer.activate(sum_h);

                auto sum_o = output_layer.sum(h);
                auto o = output_layer.activate(sum_o);

                // backpropagation
                Vector<Output> d_L_d_o = (labels[i] - o) * -2;

                // output layer
                Vector<Output> d_o_d_b{};
                std::transform(
                        sum_o.begin(), sum_o.end(), d_o_d_b.begin(),
                        [this](const auto &s) { return output_layer.activation->derivative(s); }
                );
                array<Vector<Hidden>, Output> d_o_d_w{};
                std::transform(
                        d_o_d_b.begin(), d_o_d_b.end(), d_o_d_w.begin(),
                        [h](const auto &d_o_d_b_i) { return h * d_o_d_b_i; }
                );
                array<Vector<Hidden>, Output> d_o_d_h{};
                std::transform(
                        output_layer.weights.begin(), output_layer.weights.end(),
                        d_o_d_b.begin(), d_o_d_h.begin(), std::multiplies{}
                );

                // hidden layer
                Vector<Hidden> d_h_d_b{};
                std::transform(
                        sum_h.begin(), sum_h.end(), d_h_d_b.begin(),
                        [this](const auto &s) { return hidden_layer.activation->derivative(s); }
                );
                array<Vector<Input>, Hidden> d_h_d_w{};
                std::transform(
                        d_h_d_b.begin(), d_h_d_b.end(), d_h_d_w.begin(),
                        [data_i = data[i]](const auto &d_h_d_b_i) { return data_i * d_h_d_b_i; }
                );

                // update weights and biases
                for (int j = 0; j < Output; ++j) {
                    auto o_j = d_L_d_o[j];

                    output_layer.biases[j] -= learning_rate * o_j * d_o_d_b[j];
                    for (int k = 0; k < Hidden; ++k) {
                        output_layer.weights[j][k] -= learning_rate * o_j * d_o_d_w[j][k];
                    }

                    for (int k = 0; k < Hidden; ++k) {
                        auto h_k = d_o_d_h[j][k];

                        hidden_layer.biases[k] -= learning_rate * o_j * h_k * d_h_d_b[k];
                        for (int l = 0; l < Input; ++l) {
                            hidden_layer.weights[k][l] -= learning_rate * o_j * h_k * d_h_d_w[k][l];
                        }
                    }
                }
            }
        }

        template<size_t Size>
        void train(const array<Vector<Input>, Size> &data, const array<Vector<Output>, Size> &labels) {
            for (int i = 0; i < 10000; ++i) {
                epoch(data, labels);

                if (i % 10 != 0) continue;
                loss(data, labels);
            }
        }

        template<size_t Size>
        void loss(const array<Vector<Input>, Size> &data, const array<Vector<Output>, Size> &labels) {
            auto prediction = predict(data);
            for (int j = 0; j < Output; ++j) {
                auto label_i = labels | std::views::transform([j](const auto &l) { return l[j]; });
                auto prediction_i = prediction | std::views::transform([j](const auto &p) { return p[j]; });

                auto loss = Stats::mse_loss(label_i, prediction_i);
                cout << "Loss: " << loss << '\n';
            }
        }

        template<size_t Size>
        array<Vector<Output>, Size> predict(const array<Vector<Input>, Size> &data) {
            array<Vector<Output>, Size> predictions{};
            std::transform(data.begin(), data.end(), predictions.begin(), [this](const auto &d) { return (*this)(d); });
            return predictions;
        }

    private:
        Layer<Input, Hidden> hidden_layer;
        Layer<Hidden, Output> output_layer;
        float learning_rate = 0.1;
    };
}

int main() {
    using NeuralNetwork::NeuralNetwork;

    NeuralNetwork<2, 2, 1> nn;

    array<Vector<2>, 4> data = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};

    array<Vector<1>, 4> labels = {{{0}, {1}, {1}, {0}}};

    nn.train(data, labels);

    cout << "Predictions:\n";
    auto predictions = nn.predict(data);
    for (const auto &p: predictions) { cout << p << '\n'; }

    return 0;
}
