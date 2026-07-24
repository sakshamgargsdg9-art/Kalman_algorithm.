#include "PCA.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

using namespace std;

PCA::PCA(size_t dims) : target_dims(dims), components(0, 0) {}

Matrix PCA::fitTransform(const Matrix &X)
{
    size_t n = X.getRows();
    size_t d = X.getCols();

    if (target_dims > d)
    {
        throw invalid_argument("Target dimensions cannot exceed original dimensions.");
    }

    // Step 1: Compute Mean and Center the Data
    mean_vector.assign(d, 0.0);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            mean_vector[j] += X.at(i, j);
        }
    }
    for (size_t j = 0; j < d; ++j)
    {
        mean_vector[j] /= n;
    }

    Matrix centered(n, d);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            centered.at(i, j) = X.at(i, j) - mean_vector[j];
        }
    }

    // Step 2: Compute Covariance Matrix: Cov = (1 / (n - 1)) * X^T * X
    Matrix centered_T = centered.transpose();
    Matrix cov = centered_T.multiply(centered);
    double scale = 1.0 / (n - 1);
    for (size_t i = 0; i < cov.getRows(); ++i)
    {
        for (size_t j = 0; j < cov.getCols(); ++j)
        {
            cov.at(i, j) *= scale;
        }
    }

    // Step 3: Approximate Eigen-decomposition using Power Iteration
    components = Matrix(target_dims, d);
    for (size_t k = 0; k < target_dims; ++k)
    {
        vector<double> v(d, 1.0);
        double norm = 0;
        for (double val : v)
            norm += val * val;
        norm = sqrt(norm);
        for (double &val : v)
            val /= norm;

        for (int iter = 0; iter < 50; ++iter)
        {
            vector<double> next_v(d, 0.0);
            for (size_t i = 0; i < d; ++i)
            {
                for (size_t j = 0; j < d; ++j)
                {
                    next_v[i] += cov.at(i, j) * v[j];
                }
            }
            norm = 0;
            for (double val : next_v)
                norm += val * val;
            norm = sqrt(norm);
            if (norm > 0)
            {
                for (size_t i = 0; i < d; ++i)
                    next_v[i] /= norm;
            }
            v = next_v;
        }

        for (size_t j = 0; j < d; ++j)
        {
            components.at(k, j) = v[j];
        }
    }

    // Step 4: Project data onto Principal Components
    Matrix components_T = components.transpose();
    return centered.multiply(components_T);
}