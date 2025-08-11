#include <systemc.h>
#include <vector>
#include <iostream>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

SC_MODULE(MatrixMultiplier) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    const Matrix* X_ptr = nullptr;
    const Matrix* W_ptr = nullptr;
    const Vector* b_ptr = nullptr;
    Matrix* Y_ptr = nullptr;

    void multiply_process() {
        done.write(false);
        while (true) {
            do { wait(clk->posedge_event()); } while (start.read() == false);

            std::cout << "@" << sc_time_stamp() << "Krećem množenje..." << std::endl;
            
            if (!X_ptr || !W_ptr || !Y_ptr) {
                std::cerr << "IO podaci nisu postavljeni!" << std::endl;
                sc_stop(); return;
            }

            const Matrix& X = *X_ptr;
            const Matrix& W = *W_ptr;
            Matrix& Y = *Y_ptr;

            size_t seq_len = X.size();
            size_t in_feat = X[0].size();
            size_t out_feat = W.size();

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < out_feat; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < in_feat; ++k) {
                        sum += X[i][k] * W[j][k];
                    }
                    if (b_ptr) { Y[i][j] = sum + b_ptr->at(j); } 
                    else { Y[i][j] = sum; }
                    wait(clk->posedge_event()); 
                }
            }
            
            std::cout << "@" << sc_time_stamp() << "Množenje završeno." << std::endl;
            done.write(true);

            do {
                wait(clk->posedge_event());
            } while (start.read() == true);
            
            done.write(false);
            std::cout << "@" << sc_time_stamp() << "Reset i ceka se novi zadatak" << std::endl;
        }
    }

    SC_CTOR(MatrixMultiplier) {
        SC_THREAD(multiply_process);
    }
};