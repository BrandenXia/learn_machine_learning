#include <iostream>
#include <random>

#include <Eigen/Eigen>

#include "functions.hh"


using std::cout;

template<int Size>
using Vector = Eigen::Vector<float, Size>;

template<int SizeX, int SizeY>
using Matrix = Eigen::Matrix<float, SizeX, SizeY>;

namespace NeuralNetwork {
    class Activation {
    public:
        virtual float operator()(float x) = 0;

        virtual float derivative(float x) = 0;

        template<int Size>
        Vector<Size> operator()(const Vector<Size> &x) {
            return x.unaryExpr([this](float x) { return (*this)(x); });
        }

        template<int Size>
        Vector<Size> derivative(const Vector<Size> &x) {
            return x.unaryExpr([this](float x) { return derivative(x); });
        }

        virtual ~Activation() = default;
    };

    class Sigmoid : public Activation {
    public:
        constexpr float operator()(float x) final { return Functions::sigmoid(x); }

        constexpr float derivative(float x) final { return Functions::sigmoid_derivative(x); }
    };

    template<int Input, int Output>
    class Layer {
    public:
        std::shared_ptr<Activation> activation;
        Matrix<Output, Input> weights;
        Vector<Output> biases;

        explicit Layer(std::shared_ptr<Activation> acf = std::make_shared<Sigmoid>())
                : activation(std::move(acf)) {
            std::random_device rd;
            std::mt19937 gen{rd()};
            std::uniform_real_distribution<float> dist(0, 1);

            // xavier initialization for weights
            float xavier = std::sqrt(6.0f / (Input + Output));
            weights = Matrix<Output, Input>::NullaryExpr(
                    [xavier, &dist, &gen]() { return dist(gen) * 2 * xavier - xavier; });

            // random initialization for biases
            biases = Vector<Output>::NullaryExpr([&dist, &gen]() { return dist(gen); });
        }

        Vector<Output> sum(const Vector<Input> &inputs) {
            return weights * inputs + biases;
        }

        Vector<Output> activate(const Vector<Output> &sums) {
            return sums.unaryExpr([this](float x) { return (*activation)(x); });
        }

        Vector<Output> operator()(const Vector<Input> &inputs) {
            return activate(sum(inputs));
        }
    };

    template<int Input, int Hidden, int Output> requires (Input > 0, Hidden > 0)
    class NeuralNetwork {
    public:
        NeuralNetwork() : hidden_layer(), output_layer() {}

        Vector<Output> operator()(const Vector<Input> &inputs) {
            return output_layer(hidden_layer(inputs));
        }

        template<int Size>
        void epoch(const Matrix<Input, Size>
                   &data, const Matrix<Output, Size> &labels) {
            for (int i = 0; i < Size; ++i) {
                auto sum_h = hidden_layer.sum(data.col(i));
                auto h = hidden_layer.activate(sum_h);

                auto sum_o = output_layer.sum(h);
                auto o = output_layer.activate(sum_o);

                // backpropagation
                Vector<Output> d_L_d_o = (labels.col(i) - o) * -2;

                // output layer
                Vector<Output> d_o_d_b = output_layer.activation->derivative(sum_o);
                Matrix<Output, Hidden> d_o_d_w = h * d_o_d_b;
                Vector<Hidden> d_o_d_h = output_layer.weights.transpose() * d_L_d_o;

                // hidden layer
                Vector<Hidden> d_h_d_b = hidden_layer.activation->derivative(sum_h);
                Matrix<Hidden, Input> d_h_d_w = Matrix<Hidden, Input>::Zero();
                for (int j = 0; j < Input; ++j) {
                    d_h_d_w.col(j) = data.col(i) * d_h_d_b[j];
                }

                // update weights and biases
                for (int j = 0; j < Output; ++j) {
                    output_layer.biases[j] -= learning_rate * d_L_d_o[j] * d_o_d_b[j];
                    output_layer.weights.row(j) -= learning_rate * d_L_d_o[j] * d_o_d_w.row(j);
                }
                for (int j = 0; j < Hidden; ++j) {
                    hidden_layer.biases[j] -= learning_rate * d_o_d_h[j] * d_h_d_b[j];
                    hidden_layer.weights.row(j) -= learning_rate * d_o_d_h[j] * d_h_d_w.row(j);
                }
            }
        }

        template<int Size>
        void train(const Matrix<Input, Size> &data, const Matrix<Output, Size> &labels,
                   unsigned int epochs = 10000) {
            for (int i = 0; i < epochs; ++i) {
                epoch(data, labels);

                if (i % 10 != 0) continue;
                loss(data, labels);
            }
        }

        template<int Size>
        void loss(const Matrix<Input, Size> &data, const Matrix<Output, Size> &labels) {
            // mean squared error
            auto predictions = predict(data);
            auto mse = (labels - predictions).array().square().sum() / Size;
            cout << "MSE: " << mse << '\n';
        }

        template<int Size>
        Matrix<Output, Size> predict(const Matrix<Input, Size> &data) {
            Matrix<Output, Size> predictions;
            for (int i = 0; i < Size; ++i) { predictions.col(i) = (*this)(data.col(i)); }
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

    Matrix<2, 4> data;
    data << 0, 0, 1, 1,
            0, 1, 0, 1;

    Matrix<1, 4> labels;
    labels << 0, 1, 1, 0;

    nn.train(data, labels);

    cout << "Predictions:\n";
    auto predictions = nn.predict(data);
    for (const auto &p: predictions) { cout << p << '\n'; }

    return 0;
}
