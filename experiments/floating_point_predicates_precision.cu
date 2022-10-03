#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr double H_PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111;

template<class T>
__device__ __host__ inline T pow2(const T& a) {
    return a * a;
}

inline __device__ __host__ double cross(const glm::vec2& a, const glm::vec2& b) {
    return (a.x * b.y) - (a.y * b.x);
}

inline __device__ __host__ double d_angle_vectors(const glm::vec2& u, const glm::vec2& v) {
    return atan2(cross(u, v), glm::dot(u, v)); // TODO: Get the double version of this function. For now, the compiler does not find the function atan2
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
inline __device__ __host__ int angle_incircle(const glm::vec2& com_a, const glm::vec2& op1, const glm::vec2& com_b, const glm::vec2& op2) {
    glm::dvec2 u; // vector
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
    
    return fabs(alpha + beta) / H_PI - 0.0000001;
}


__device__ __host__ inline double incircle(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d) {
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
    //return (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)) > EPS;
}


__global__ void calc_predicate_incircle(int resolution, glm::dvec2 a, glm::dvec2 b, glm::dvec2 c, glm::dvec2 d, double reasonable_eps, double* dest) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx%resolution;
    const int i = idx/resolution;
    if (i < resolution) {
        glm::dvec2 perturbation((i-resolution/2) * reasonable_eps, (j-resolution/2) * reasonable_eps);
        dest[i*resolution+j] = incircle(a, b, c, d + perturbation);
    }
}

__global__ void calc_predicate_angle_incircle(int resolution, glm::dvec2 a, glm::dvec2 b, glm::dvec2 c, glm::dvec2 d, double reasonable_eps, double* dest) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (i < resolution) {
        glm::dvec2 perturbation((i - resolution / 2) * reasonable_eps, (j - resolution / 2) * reasonable_eps);
        dest[i*resolution+j] = angle_incircle(a, b, c, d + perturbation);
    }
}

__global__ void get_image_from_angle_incircle(int resolution, double* org, char* dest) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (idx < resolution*resolution) {
        //printf("idx: %d val: %lf\n", idx, org[(i * resolution + j)]);
        dest[(i+resolution*j)*3+0] = (org[(i*resolution+j)]>0)*254;
        dest[(i+resolution*j)*3+1] = (org[(i*resolution+j)]==0)*254;
        dest[(i+resolution*j)*3+2] = (org[(i*resolution+j)]<0)*254;
    }
}

__global__ void get_image_from_incircle(int resolution, double* org, char* dest) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % resolution;
    const int i = idx / resolution;
    if (idx < resolution*resolution) {
        //printf("idx: %d val: %lf\n", idx, org[(i * resolution + j)]);
        dest[(i+resolution*j)*3+0] = (org[(i*resolution+j)]>0)*254;
        dest[(i+resolution*j)*3+1] = (org[(i*resolution+j)]==0)*254;
        dest[(i+resolution*j)*3+2] = (org[(i*resolution+j)]<0)*254;
    }
}

int main(int argc, char* argv[]) {

    glm::dvec2 a(-1.0, 0.0), b(0.0, -1.0), c(1.0, 0.0), d(0.0, 1.0);

    double significand_x;
    int exponent_x;
    significand_x = frexp(d.x, &exponent_x);

    double reasonable_eps = pow(2.0, (double)(exponent_x - 30));
    if (argc > 3 && atoi(argv[3]) > 0) {
        // max -53
        reasonable_eps = pow(2.0, (double)(exponent_x - atoi(argv[3])));
    }
    printf("Perturbation used: %lf\n", reasonable_eps);


    int resolution = 16384;
    if (argc > 1 && atoi(argv[1]) > 0){
        resolution = atoi(argv[1]);
    }
    double* d_predicates, * h_predicates = new double[resolution * resolution];
    cudaMalloc((void**)&d_predicates, resolution * resolution * sizeof(double));
    char *d_image, *h_image = new char[3*resolution*resolution];
    cudaMalloc((void**)&d_image, resolution * resolution * 3 * sizeof(char));

    const int blocksize = 128;
    dim3 dimBlock(blocksize);
    dim3 dimGrid((resolution*resolution + blocksize - 1) / dimBlock.x);
    
    if (argc>2 && atoi(argv[2]) > 0) {
        printf("Writting incircle\n");
        calc_predicate_incircle << <dimGrid, dimBlock >> > (resolution, a, b, c, d, reasonable_eps, d_predicates);
        cudaDeviceSynchronize();
        get_image_from_incircle << <dimGrid, dimBlock >> > (resolution, d_predicates, d_image);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image, d_image, resolution * resolution * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

    }
    else {
        printf("Writting angle incircle\n");
        calc_predicate_angle_incircle << <dimGrid, dimBlock >> > (resolution, a, b, c, d, reasonable_eps, d_predicates);
        cudaDeviceSynchronize();
        get_image_from_angle_incircle << <dimGrid, dimBlock >> > (resolution, d_predicates, d_image);
        cudaDeviceSynchronize();
        cudaMemcpy(h_image, d_image, resolution * resolution * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

    }

    stbi_flip_vertically_on_write(true);

    if (argc > 4) {
        stbi_write_bmp(argv[4], resolution, resolution, 3, h_image);
    }
    else stbi_write_bmp("res.bmp", resolution, resolution, 3, h_image);
    return 0;
}