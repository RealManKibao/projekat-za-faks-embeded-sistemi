#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Potrebno za automatsku konverziju std::vector

// Deklaracije tipova i pomoćne funkcije idu ovde, na početku
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// --------------------------------------------------------------------
// OVDE IDE VAŠA GLAVNA C++ FUNKCIJA KOJU ŽELITE DA POZIVATE IZ PYTHONA
// --------------------------------------------------------------------
Matrix mnozenje_matrica_realtime(const Matrix& X, const Matrix& W, const Vector& b) {
    
    if (X.empty() || W.empty() || b.empty()) {
        throw std::runtime_error("Ulazne matrice ne smeju biti prazne!");
    }

    size_t seq_len = X.size();
    size_t in_feat = X[0].size();
    size_t out_feat = W.size();

    // Provera dimenzija za svaki slučaj
    if (W[0].size() != in_feat || b.size() != out_feat) {
        throw std::runtime_error("Dimenzije matrica se ne poklapaju!");
    }

    Matrix Y(seq_len, Vector(out_feat, 0.0));

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < out_feat; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < in_feat; ++k) {
                sum += X[i][k] * W[j][k];
            }
            Y[i][j] = sum + b[j];
        }
    }
    
    return Y;
}

// --------------------------------------------------------------------
// "PRIKLJUČAK" ZA PYTHON IDE NA KRAJ FAJLA. NEMA MAIN() FUNKCIJE!
// --------------------------------------------------------------------
namespace py = pybind11;

PYBIND11_MODULE(moj_akcelerator, m) {
    m.doc() = "Moj C++ modul za množenje matrica, ubrzan hardverom!";
    
    // Ovde "kažemo" Pythonu: 
    // Funkcija u Pythonu će se zvati "mnozenje_matrica".
    // Ona zapravo poziva našu C++ funkciju "&mnozenje_matrica_realtime".
    // Treći argument je kratak opis funkcije.
    m.def("mnozenje_matrica", &mnozenje_matrica_realtime, "Funkcija koja mnozi matrice u C++");
}