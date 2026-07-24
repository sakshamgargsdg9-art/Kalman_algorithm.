#include <iostream>
#include "Matrix.hpp"
#include "PCA.hpp"
#include "KMeans.hpp"

using namespace std;

int main()
{
    try
    {
        cout << "=== Data Clustering & Dimensionality Reduction Tool ===" << endl;

        Matrix X({{1.0, 2.0, 1.5, 0.8},
                  {1.2, 1.9, 1.6, 0.9},
                  {8.0, 7.8, 8.2, 7.9},
                  {8.5, 8.1, 7.9, 8.2},
                  {1.1, 2.1, 1.4, 0.7}});

        cout << "\nOriginal Data Matrix (Dimensions: " << X.getRows() << "x" << X.getCols() << "):" << endl;
        X.print();

        size_t target_dims = 2;
        PCA pca(target_dims);
        Matrix reduced_X = pca.fitTransform(X);

        cout << "\nReduced Data Matrix after PCA (Dimensions: " << reduced_X.getRows() << "x" << reduced_X.getCols() << "):" << endl;
        reduced_X.print();

        int k = 2;
        KMeans kmeans(k);
        vector<int> cluster_labels = kmeans.fitPredict(reduced_X);

        cout << "\nK-Means Cluster Assignments:" << endl;
        for (size_t i = 0; i < cluster_labels.size(); ++i)
        {
            cout << "Sample " << i << " -> Cluster " << cluster_labels[i] << endl;
        }
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}