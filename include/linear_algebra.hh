#ifndef MACHINE_LEARNING_LINEAR_ALGEBRA_HH
#define MACHINE_LEARNING_LINEAR_ALGEBRA_HH

#include <array>


namespace LinearAlgebra {
    template<size_t N>
    class Vector : public std::array<float, N> {
    public:
        Vector<N> operator+(const Vector<N> &other) const;

        Vector<N> operator-(const Vector<N> &other) const;

        Vector<N> operator*(float scalar) const;

        float dot(const Vector<N> &other) const;
    };

    template<size_t N>
    Vector<N> Vector<N>::operator+(const Vector<N> &other) const {
        Vector<N> result{};
        std::transform(this->begin(), this->end(), other.begin(), result.begin(), std::plus{});
        return result;
    }

    template<size_t N>
    Vector<N> Vector<N>::operator-(const Vector<N> &other) const {
        Vector<N> result{};
        std::transform(this->begin(), this->end(), other.begin(), result.begin(), std::minus{});
        return result;
    }

    template<size_t N>
    Vector<N> Vector<N>::operator*(float scalar) const {
        Vector<N> result{};
        std::transform(this->begin(), this->end(), result.begin(), [scalar](const auto &x) {
            return x * scalar;
        });
        return result;
    }

    template<size_t N>
    float Vector<N>::dot(const Vector<N> &other) const {
        return std::inner_product(this->begin(), this->end(), other.begin(), 0.0f);
    }

    template<size_t N>
    std::ostream &operator<<(std::ostream &os, const Vector<N> &v) {
        os << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<float>(os, ", "));
        return os << "\b\b]";
    }
}

#endif //MACHINE_LEARNING_LINEAR_ALGEBRA_HH
