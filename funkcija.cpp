#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
namespace py = pybind11;


//Množenje matrica
Matrix mnozenje_matrica_internal(const Matrix& X, const Matrix& W, const Vector& b, bool has_bias) {
    if (X.empty() || W.empty() || X[0].empty() || W[0].empty()) {
        throw std::runtime_error("Ulazne matrice ne smeju biti prazne!");
    }
    size_t seq_len = X.size();
    size_t in_feat = X[0].size();
    size_t out_feat = W.size();
    if (W[0].size() != in_feat || (has_bias && b.size() != out_feat)) {
        throw std::runtime_error("Dimenzije matrica se ne poklapaju!");
    }
    Matrix Y(seq_len, Vector(out_feat, 0.0));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < out_feat; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < in_feat; ++k) {
                sum += X[i][k] * W[j][k];
            }
            Y[i][j] = sum + (has_bias ? b[j] : 0.0);
        }
    }
    return Y;
}

//Softmax funkcija
Matrix softmax_internal(const Matrix& mat) {
    Matrix result = mat;
    for (size_t i = 0; i < mat.size(); ++i) {
        double max_val = mat[i][0];
        for (size_t j = 1; j < mat[i].size(); ++j) { if (mat[i][j] > max_val) max_val = mat[i][j]; }
        double sum_exp = 0.0;
        for (size_t j = 0; j < mat[i].size(); ++j) {
            result[i][j] = std::exp(mat[i][j] - max_val);
            sum_exp += result[i][j];
        }
        for (size_t j = 0; j < mat[i].size(); ++j) { result[i][j] /= sum_exp; }
    }
    return result;
}

//Glavna funkcija za Attention block koja se izvovi u pajton kod
Matrix attention_realtime(const Matrix& Q, const Matrix& K, const Matrix& V) {
    //Množenje Q * K^T
    Matrix scores = mnozenje_matrica_internal(Q, K, {}, false);

    //Skaliranje
    double d_k = static_cast<double>(Q[0].size());
    double scale_factor = 1.0 / sqrt(d_k);
    for (auto& row : scores) { for (auto& val : row) { val *= scale_factor; } }

    //Softmax
    Matrix probs = softmax_internal(scores);

    //Množenje sa V
    Matrix V_T(V[0].size(), Vector(V.size()));
    for(size_t i=0; i < V.size(); ++i) { for(size_t j=0; j < V[0].size(); ++j) { V_T[j][i] = V[i][j]; } }
    Matrix final_result = mnozenje_matrica_internal(probs, V_T, {}, false);
    return final_result;
}

//Ovo je deo gde linkujem sa pajton kodom
PYBIND11_MODULE(moja_realizacija_funkcije, m) {
    m.doc() = "Moj C++ modul za Attention block";
    m.def("attention", &attention_realtime, "Kompletna 'scaled dot-product attention' operacija u C++");
}