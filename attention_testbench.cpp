#include <systemc.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "attention_module.h" //Uklučuje se glavni modul

//Funkcije za čitanje i pisanje
Matrix readMatrix(const std::string& filename);
Vector readVector(const std::string& filename);
void writeMatrix(const std::string& filename, const Matrix& mat);


SC_MODULE(Testbench) {
    sc_clock clk;
    sc_signal<bool> start_sig;
    sc_signal<bool> done_sig;

    AttentionModule uut; //Instanca mog Attention modela

    Matrix Q_data, K_data, V_data, Y_data;

    void stimulus_process() {
        std::cout << "Učitavam Q, K, V vektore..." << std::endl;
        Q_data = readMatrix("ulaz_Q.txt");
        K_data = readMatrix("ulaz_K.txt");
        V_data = readMatrix("ulaz_V.txt");
        
        //Inicijalizujem izlaznu matricu
        if (!Q_data.empty() && !V_data.empty()) {
            Y_data.resize(Q_data.size(), Vector(V_data[0].size()));
        }

        //Povezujem podatke sa Attention modulom
        uut.Q_ptr = &Q_data;
        uut.K_ptr = &K_data;
        uut.V_ptr = &V_data;
        uut.Y_ptr = &Y_data;

        wait(10, SC_NS);
        std::cout << "Postavljam START signal." << std::endl;
        start_sig.write(true);

        wait(done_sig.posedge_event());
        std::cout << "Signal je primljen." << std::endl;
        start_sig.write(false);
        wait(1, SC_NS);

        std::cout << "Upisujem rezultat u fajl 'izlaz_Attention_systemc.txt'..." << std::endl;
        writeMatrix("izlaz_Attention_systemc.txt", Y_data);

        std::cout << "Simulacija završena." << std::endl;
        sc_stop();
    }

    SC_CTOR(Testbench) : clk("clk", 10, SC_NS), uut("AttentionModule_1") {
        uut.clk(clk);
        uut.start(start_sig);
        uut.done(done_sig);
        SC_THREAD(stimulus_process);
    }
};

int sc_main(int argc, char* argv[]) {
    Testbench tb("Testbench_1");
    sc_start();
    return 0;
}


//Implementacija funkcije za čitanje i pisanje
Matrix readMatrix(const std::string& filename) {
    Matrix mat;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double val;
        while (ss >> val) { 
            row.push_back(val); 
        }
        if (!row.empty()) {
            mat.push_back(row);
        }
    }
    return mat;
}

Vector readVector(const std::string& filename) {
    Vector vec;
    std::ifstream file(filename);
    double val;
    while (file >> val) { 
        vec.push_back(val); 
    }
    return vec;
}

void writeMatrix(const std::string& filename, const Matrix& mat) {
    std::ofstream file(filename);
    for (const auto& row : mat) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i] << (i == row.size() - 1 ? "" : " ");
        }
        file << std::endl;
    }
}