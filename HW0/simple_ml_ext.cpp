#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
void mat_exp(float *Z, int r, int c)
{
    for (size_t i = 0; i < r * c; ++i)
        Z[i] = exp(Z[i]);
}

void dot_mul(const float *X, const float *Y, float *Z, int m, int n, int k)
{
    // X: m x n, Y: n x k, Z: m x k
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < k; j++)
        {
            Z[i * k + j] = 0;
            for (size_t s = 0; s < n; s++)
                Z[i * k + j] += X[i * n + s] * Y[s * k + j];
        }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    auto T = (m + batch - 1) / batch;
    for (size_t iter = 0; iter < T; ++iter)
    {
        const float *tmpX = &X[iter * batch * n];
        // Z = normalize(softmax(X @ theta))
        float *Z = new float[batch * k]; // batch * k
        dot_mul(tmpX, theta, Z, batch, n, k);
        mat_exp(Z, batch, k);
        for (size_t r = 0; r < batch; ++r)
        {
            float s = 0;
            for (size_t c = 0; c < k; ++c)
                s += Z[r * k + c];
            for (size_t c = 0; c < k; ++c)
                Z[r * k + c] /= s;
        }
        // Iy = One-hot(y)
        float *Iy = new float[batch * k];
        for (size_t i = 0; i < batch; ++i)
            for (size_t j = 0; j < k; ++j)
                Iy[i * k + j] = y[iter * batch + i] == j ? 1 : 0;
        for (size_t i = 0; i < batch; ++i)
            for (size_t j = 0; j < k; ++j)
                Z[i * k + j] -= Iy[i * k + j];
        float *x_T = new float[n * batch];
        float *grad = new float[n * k];
        for (size_t i = 0; i < batch; i++)
            for (size_t j = 0; j < n; j++)
                x_T[j * batch + i] = tmpX[i * n + j];
        dot_mul(x_T, Z, grad, n, batch, k);
        for (size_t i = 0; i < n * k; i++)
            theta[i] -= lr / batch * grad[i]; // SGD update
        delete[] Z;
        delete[] x_T;
        delete[] grad;
        delete[] Iy;
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta, float lr, int batch) {
            softmax_regression_epoch_cpp(static_cast<const float *>(X.request().ptr),
                                         static_cast<const unsigned char *>(y.request().ptr),
                                         static_cast<float *>(theta.request().ptr), X.request().shape[0],
                                         X.request().shape[1], theta.request().shape[1], lr, batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
