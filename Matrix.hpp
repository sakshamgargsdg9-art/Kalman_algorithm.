#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <string>

using vector_2d = std::vector<std::vector<double>>;
using vector_1d = std::vector<double>;

class Matrix
{
private:
    vector_2d data;
    size_t rows;
    size_t cols;

public:
    Matrix(size_t r, size_t c, double initial_val = 0.0);
    Matrix(const vector_2d &d);

    size_t getRows() const;
    size_t getCols() const;

    double &at(size_t r, size_t c);
    const double &at(size_t r, size_t c) const;

    vector_1d &operator[](size_t r);
    const vector_1d &operator[](size_t r) const;

    Matrix transpose() const;
    Matrix multiply(const Matrix &other) const;

    static Matrix loadFromCSV(const std::string &filepath);
    void print() const;
};

#endif