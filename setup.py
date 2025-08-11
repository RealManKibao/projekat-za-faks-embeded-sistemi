import sys
from setuptools import setup, Extension
import pybind11

#definicija c++ modula
ext_modules = [
    Extension(
        'moja_realizacija_funkcije',                #Ime kako ce se importovati u pajtonu
        ['funkcija.cpp'],        #Ovo je moj c++ fajl koji sam pisao za dati algoritam
        include_dirs=[
            pybind11.get_include(),       #Automatski dodajemo pybind11 putanju
        ],
        language='c++',
        #Ovde dodajemo flag da kompajler koristi C++17 standard
        extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    ),
]

setup(
    name='moja_realizacija_funkcije',
    version='1.0.0',
    author='Veljko',
    description='C++ funkcija za Whisper Attention',
    ext_modules=ext_modules,
)