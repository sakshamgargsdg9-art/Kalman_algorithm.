#include "Matrix.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;

Matrix::Matrix(size_t r, size_t c, double initial_val) : rows(r), cols(c), data(r, vector<double>(c, initial_val)) {}

Matrix::Matrix(const vector<vector<double>> &d) : data(d), rows(d.size()), cols(d.empty() ? 0 : d[0].size()) {}

size_t Matrix::getRows() const { return rows; }
size_t Matrix::getCols() const { return cols; }

double &Matrix::at(size_t r, size_t c) { return data[r][c]; }
const double &Matrix::at(size_t r, size_t c) const { return data[r][c]; }

vector<double> &Matrix::operator[](size_t r) { return data[r]; }
const vector<double> &Matrix::operator[](size_t r) const { return data[r]; }

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result.at(j, i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw invalid_argument("Matrix dimension mismatch for multiplication.");
    }
    Matrix result(rows, other.cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t k = 0; k < cols; ++k)
        {
            for (size_t j = 0; j < other.cols; ++j)
            {
                result.at(i, j) += data[i][k] * other.at(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::loadFromCSV(const string &filepath)
{
    ifstream file(filepath);
    if (!file.is_open())
    {
        throw runtime_error("Could not open file: " + filepath);
    }

    vector<vector<double>> parsedData;
    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        string val;
        vector<double> row;
        while (getline(ss, val, ','))
        {
            row.push_back(stod(val));
        }
        if (!row.empty())
        {
            parsedData.push_back(row);
        }
    }
    return Matrix(parsedData);
}

void Matrix::print() const
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            cout << data[i][j] << "\t";
        }
        cout << "\n";
    }
}