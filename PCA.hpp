#ifndef PCA_HPP
#define PCA_HPP

#include "Matrix.hpp"
#include <vector>

class PCA
{
private:
    size_t target_dims;
    Matrix components;
    std::vector<double> mean_vector;

public:
    PCA(size_t dims);
    Matrix fitTransform(const Matrix &X);
};

#endif