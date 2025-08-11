#include <systemc.h>
#include <vector>
#include <iostream>
#include <cmath> 
#include "matrix_multiplier.h"

Matrix softmax(const Matrix& mat) {
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

SC_MODULE(AttentionModule) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    const Matrix* Q_ptr = nullptr;
    const Matrix* K_ptr = nullptr;
    const Matrix* V_ptr = nullptr;
    Matrix* Y_ptr = nullptr;

    MatrixMultiplier mat_mul_unit;
    sc_signal<bool> mat_mul_start_sig;
    sc_signal<bool> mat_mul_done_sig;

    Matrix scores;
    Matrix probs;

    void attention_process() {
        done.write(false);
        while (true) {
            wait(start.posedge_event());
            std::cout << "@" << sc_time_stamp() << "Start signal je primljen." << std::endl;

            //Q * K^T
            std::cout << "@" << sc_time_stamp() << "Počinjem Q * K^T..." << std::endl;
            scores.assign(Q_ptr->size(), Vector(K_ptr->size()));
            mat_mul_unit.X_ptr = Q_ptr; mat_mul_unit.W_ptr = K_ptr; mat_mul_unit.b_ptr = nullptr; mat_mul_unit.Y_ptr = &scores;
            
            mat_mul_start_sig.write(true);
            wait(mat_mul_done_sig.posedge_event());
            mat_mul_start_sig.write(false);
            wait(clk->posedge_event());
            std::cout << "@" << sc_time_stamp() << "Q * K^T završeno." << std::endl;

            //Skaliranje
            double d_k = static_cast<double>(Q_ptr->at(0).size());
            double scale_factor = 1.0 / sqrt(d_k);
            for (auto& row : scores) { for (auto& val : row) { val *= scale_factor; } }
            std::cout << "@" << sc_time_stamp() << "Skaliranje završeno." << std::endl;

            //Softmax
            probs = softmax(scores);
            std::cout << "@" << sc_time_stamp() << "Softmax završen." << std::endl;

            //Množenje sa V
            std::cout << "@" << sc_time_stamp() << "Počinjem Probs * V..." << std::endl;
            Matrix V_T(V_ptr->at(0).size(), Vector(V_ptr->size()));
            for(size_t i=0; i < V_ptr->size(); ++i) { for(size_t j=0; j < V_ptr->at(0).size(); ++j) { V_T[j][i] = V_ptr->at(i)[j]; } }
            mat_mul_unit.X_ptr = &probs; mat_mul_unit.W_ptr = &V_T; mat_mul_unit.b_ptr = nullptr; mat_mul_unit.Y_ptr = Y_ptr;

            mat_mul_start_sig.write(true);
            wait(mat_mul_done_sig.posedge_event());
            mat_mul_start_sig.write(false);
            wait(clk->posedge_event());
            std::cout << "@" << sc_time_stamp() << "Probs * V završeno." << std::endl;

            //Ovde javljamo testbenchu da je ceo Attention proces gotov...
            done.write(true);
            wait(clk->posedge_event());
            done.write(false);
        }
    }

    SC_CTOR(AttentionModule) : mat_mul_unit("MatMul_Worker") {
        SC_THREAD(attention_process);
        mat_mul_unit.clk(clk);
        mat_mul_unit.start(mat_mul_start_sig);
        mat_mul_unit.done(mat_mul_done_sig);
    }
};