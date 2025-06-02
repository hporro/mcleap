#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/vec2.hpp>
#include <random>

#include <math.h>

#include "predicates.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

constexpr double H_PI = 3.14159265358979323846;

template <class T>
__device__ __host__ inline T pow2(const T &a)
{
    return a * a;
}

inline __device__ __host__ double cross(const glm::dvec2 &a, const glm::dvec2 &b)
{
    return (a.x * b.y) - (a.y * b.x);
}

inline __device__ __host__ double dot(const glm::dvec2 &a, const glm::dvec2 &b)
{
    return (a.x * b.x) + (a.y * b.y);
}

inline __device__ __host__ double d_angle_vectors(const glm::dvec2 &u, const glm::dvec2 &v)
{
    return atan2(abs(cross(u, v)), dot(u, v));
}

// taken from https://gitlab.com/hporro01/mcleap/-/blob/main/src/kernels.cuh
///         com_a
///           +
///          /|\
///         / | \
///        /  |  \
///       /   |   \
///      / \  |  / \
/// op1 + α | | | β + op2
///      \ /  |  \ /
///       \   |   /
///        \  |  /
///         \ | /
///          \|/
///           +
///         com_b
/// Computes wether or not we have to flip (either 0 or 1). It is 1 if α+β>PI+EPS
inline __device__ __host__ double angle_incircle(const glm::dvec2 &com_a, const glm::dvec2 &op1, const glm::dvec2 &com_b, const glm::dvec2 &op2)
{
    glm::dvec2 u;    // vector
    glm::dvec2 p, q; // points
    // get two vectors of the first triangle
    p = op1;
    q = com_a;
    u = q - p; //! + 5 flop
    q = com_b;
    double alpha = d_angle_vectors(u, q - p);
    // the same for other triangle
    p = op2;
    q = com_a;
    u = q - p;
    q = com_b;
    double beta = d_angle_vectors(u, q - p);

    return fabs(alpha + beta) / H_PI;
}

__device__ __host__ inline double matrix_incircle(const glm::dvec2 &a, const glm::dvec2 &b, const glm::dvec2 &c, const glm::dvec2 &d)
{
    const double a00 = a.x - d.x;
    const double a01 = a.y - d.y;
    const double a02 = pow2(a00) + pow2(a01);
    const double a10 = b.x - d.x;
    const double a11 = b.y - d.y;
    const double a12 = pow2(a10) + pow2(a11);
    const double a20 = c.x - d.x;
    const double a21 = c.y - d.y;
    const double a22 = pow2(a20) + pow2(a21);

    double det = (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20));

    return det;
    // return (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)) > EPS;
}

__host__ inline double exact_incircle(const glm::dvec2 &a, const glm::dvec2 &b, const glm::dvec2 &c, const glm::dvec2 &d)
{
    double a_arr[2] = {a.x, a.y};
    double b_arr[2] = {b.x, b.y};
    double c_arr[2] = {c.x, c.y};
    double d_arr[2] = {d.x, d.y};

    return incircle(a_arr, b_arr, c_arr, d_arr);
}

#define EPS 1e-10

__global__ void calc_predicate_incircle(int resolution, glm::dvec2 a, glm::dvec2 b, glm::dvec2 c, glm::dvec2 d, double reasonable_eps, double *dest)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (i < resolution)
    {
        glm::dvec2 perturbation((i - resolution / 2) * reasonable_eps, (j - resolution / 2) * reasonable_eps);
        dest[i * resolution + j] = matrix_incircle(a, b, c, d + perturbation);
    }
}

#undef EPS

__global__ void calc_predicate_angle_incircle(int resolution, glm::dvec2 a, glm::dvec2 b, glm::dvec2 c, glm::dvec2 d, double reasonable_eps, double *dest)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (i < resolution)
    {
        glm::dvec2 perturbation((i - resolution / 2) * reasonable_eps, (j - resolution / 2) * reasonable_eps);
        dest[i * resolution + j] = angle_incircle(a, b, c, d + perturbation);
    }
}

__global__ void get_image_from_angle_incircle(int resolution, double *org, char *dest)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (idx < resolution * resolution)
    {
        // printf("idx: %d val: %lf\n", idx, org[(i * resolution + j)]);
        dest[(i + resolution * j) * 3 + 0] = (org[(i * resolution + j)] > 1) * 254;
        dest[(i + resolution * j) * 3 + 1] = (org[(i * resolution + j)] == 1) * 254;
        dest[(i + resolution * j) * 3 + 2] = (org[(i * resolution + j)] < 1) * 254;
    }
}

__global__ void get_image_from_incircle(int resolution, double *org, char *dest)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (idx < resolution * resolution)
    {
        // printf("idx: %d val: %lf\n", idx, org[(i * resolution + j)]);
        dest[(i + resolution * j) * 3 + 0] = (org[(i * resolution + j)] > 0) * 254;
        dest[(i + resolution * j) * 3 + 1] = (org[(i * resolution + j)] == 0) * 254;
        dest[(i + resolution * j) * 3 + 2] = (org[(i * resolution + j)] < 0) * 254;
    }
}

glm::dvec2 rotate2d(const glm::dvec2 &a, double angle)
{
    double c = cos(angle);
    double s = sin(angle);
    return glm::dvec2{a.x * c - a.y * s, a.x * s + a.y * c};
}

int main(int argc, char *argv[])
{

    // args -> resolution, incircle matrix?, eps, rotation

    glm::dvec2 a(-1.0, 0.0), b(0.0, -1.0), c(1.0, 0.0), d(0.0, 1.0);

    double rotation = 0.0;
    if (argc > 5)
    {
        rotation = atof(argv[5]);
    }

    a = rotate2d(a, H_PI * rotation);
    b = rotate2d(b, H_PI * rotation);
    c = rotate2d(c, H_PI * rotation);
    d = rotate2d(d, H_PI * rotation);

    double reasonable_eps = 2.0 * 1e-7; // default reasonable eps
    int exponent_x = 23;                // default exponent for 32 bit float

    if (argc > 3 && atoi(argv[3]) > 0)
    {
        // max -53
        printf("Using exponentx %d\n", atoi(argv[3]));
        exponent_x = atoi(argv[3]);
        // reasonable_eps = 1.0 / (1 << atoi(argv[3]));
        reasonable_eps = pow(2.0, -exponent_x);
    }
    printf("Exponent found: %d Perturbation used: %Lf\n", exponent_x, reasonable_eps);

    int resolution = 16384;
    if (argc > 1 && atoi(argv[1]) > 0)
    {
        resolution = atoi(argv[1]);
    }
    double *d_predicates, *h_predicates = new double[resolution * resolution];
    cudaMalloc((void **)&d_predicates, resolution * resolution * sizeof(double));
    char *d_image, *h_image = new char[3 * resolution * resolution];
    cudaMalloc((void **)&d_image, resolution * resolution * 3 * sizeof(char));

    const int blocksize = 128;
    dim3 dimBlock(blocksize);
    dim3 dimGrid((resolution * resolution + blocksize - 1) / dimBlock.x);

    if (argc > 2 && atoi(argv[2]) == 1)
    {
        printf("Writting matrix_incircle\n");
        calc_predicate_incircle<<<dimGrid, dimBlock>>>(resolution, a, b, c, d, reasonable_eps, d_predicates);
        cudaDeviceSynchronize();
        get_image_from_incircle<<<dimGrid, dimBlock>>>(resolution, d_predicates, d_image);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image, d_image, resolution * resolution * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
    else if (argc > 2 && atoi(argv[2]) == 2)
    {
        printf("Writting exact_incircle\n");
        for (int i = 0; i < resolution * resolution; i++)
        {
            int j = i % resolution;
            int k = i / resolution;
            glm::dvec2 perturbation((k - resolution / 2) * reasonable_eps, (j - resolution / 2) * reasonable_eps);
            h_predicates[i] = exact_incircle(a, b, c, d + perturbation);
        }
        cudaMemcpy(d_predicates, h_predicates, resolution * resolution * sizeof(double), cudaMemcpyHostToDevice);
        get_image_from_incircle<<<dimGrid, dimBlock>>>(resolution, d_predicates, d_image);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image, d_image, resolution * resolution * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
    else
    {
        printf("Writting angle incircle\n");
        calc_predicate_angle_incircle<<<dimGrid, dimBlock>>>(resolution, a, b, c, d, reasonable_eps, d_predicates);
        cudaDeviceSynchronize();
        get_image_from_angle_incircle<<<dimGrid, dimBlock>>>(resolution, d_predicates, d_image);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image, d_image, resolution * resolution * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    stbi_flip_vertically_on_write(true);

    if (argc > 4)
    {
        stbi_write_png(argv[4], resolution, resolution, 3, h_image, 0);
    }
    else
        stbi_write_png("res.bmp", resolution, resolution, 3, h_image, 0);
    return 0;
}