#ifndef KMEANS_HPP
#define KMEANS_HPP

#include "Matrix.hpp"
#include <vector>

class KMeans
{
private:
    int k;
    int max_iterations;
    Matrix centroids;

    double euclideanDistance(const std::vector<double> &a, const std::vector<double> &b);

public:
    KMeans(int clusters, int max_iter = 100);
    std::vector<int> fitPredict(const Matrix &X);
    Matrix getCentroids() const;
};

#endif