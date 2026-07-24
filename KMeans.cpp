#include "KMeans.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>

using namespace std;

KMeans::KMeans(int clusters, int max_iter) : k(clusters), max_iterations(max_iter), centroids(0, 0) {}

double KMeans::euclideanDistance(const vector<double> &a, const vector<double> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

vector<int> KMeans::fitPredict(const Matrix &X)
{
    size_t n = X.getRows();
    size_t d = X.getCols();
    srand(42);

    vector<vector<double>> init_centroids(k, vector<double>(d));
    for (int i = 0; i < k; ++i)
    {
        int idx = rand() % n;
        for (size_t j = 0; j < d; ++j)
        {
            init_centroids[i][j] = X.at(idx, j);
        }
    }
    centroids = Matrix(init_centroids);

    vector<int> labels(n, -1);
    bool changed = true;
    int iter = 0;

    while (changed && iter < max_iterations)
    {
        changed = false;
        vector<int> new_labels(n);

        for (size_t i = 0; i < n; ++i)
        {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;

            vector<double> point(d);
            for (size_t j = 0; j < d; ++j)
                point[j] = X.at(i, j);

            for (int c = 0; c < k; ++c)
            {
                vector<double> cent(d);
                for (size_t j = 0; j < d; ++j)
                    cent[j] = centroids.at(c, j);

                double dist = euclideanDistance(point, cent);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            new_labels[i] = best_cluster;
            if (new_labels[i] != labels[i])
            {
                changed = true;
            }
        }
        labels = new_labels;

        vector<vector<double>> new_centroids(k, vector<double>(d, 0.0));
        vector<int> counts(k, 0);

        for (size_t i = 0; i < n; ++i)
        {
            int cluster = labels[i];
            counts[cluster]++;
            for (size_t j = 0; j < d; ++j)
            {
                new_centroids[cluster][j] += X.at(i, j);
            }
        }

        for (int c = 0; c < k; ++c)
        {
            if (counts[c] > 0)
            {
                for (size_t j = 0; j < d; ++j)
                {
                    new_centroids[c][j] /= counts[c];
                }
            }
        }
        centroids = Matrix(new_centroids);
        iter++;
    }

    return labels;
}

Matrix KMeans::getCentroids() const
{
    return centroids;
}