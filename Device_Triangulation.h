#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <stack>

#include <glm/glm.hpp>
#include <glm/vec2.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>


#include "Common_Triangulation.h"
#include "Helpers_Triangulation.h"

#include "kernels.cuh"

// -------------------------------------------
// Structs defined
// -------------------------------------------

struct DeviceTriangulation;


// this implementation does not handle removing defragmentation of the std::vectors
struct DeviceTriangulation {
    // -------------------------------------------
    // host triangulation pointer
    HostTriangulation* h_triangulation;
    // main device data
    int m_pos_size;
    int m_v_size;
    int m_he_size;
    int m_t_size;

    glm::vec2* m_pos;
    Vertex* m_v;
    HalfEdge* m_he;
    Triangle* m_t;
    // helper device data
    int blocksize = 128;
    int* m_flag;
    int* m_helper_v;
    int* m_helper_he;
    int* m_helper_t;

    // -------------------------------------------
    // constructors & destructors
    DeviceTriangulation(HostTriangulation* h_triangulation);
    ~DeviceTriangulation();

    // -------------------------------------------
    // memory transfer helpers. It transfers the memory from the host to the device or vice versa
    bool transferToHost();
    bool transferToDevice();
    bool allocateMemory();
    bool freeMemory();

    // -------------------------------------------
    // triangle finding
    __device__ __host__ int findContainingTriangleIndexCheckingAll(glm::vec2 p);
    __device__ __host__ int findContainingTriangleIndexWalking(glm::vec2 p, int t_index);

    // -------------------------------------------
    // delonizing
    bool delonize();

    // -------------------------------------------
    // Moving points
    bool movePoints(glm::vec2* d); //moves without fixing anything
    bool moveFlipflop(int p_index, glm::vec2 d); //removes and reinserts a point
    bool moveFlipflopDelaunay(int p_index, glm::vec2 d); //removes and reinserts a point, and then Delonizates
    bool moveUntangling(int p_index, glm::vec2 d); //moves without fixing anything, and then untangles
    bool moveUntanglingDelaunay(int p_index, glm::vec2 d); //moves without fixing anything, and then untangles, and then Delonizates

    // -------------------------------------------
    // Untangling Mesh
    // based on Shewchuk's untangling
    bool untangle();

    // -------------------------------------------
    // Neighborhood searching
    template<int maxRingSize>
    bool oneRing(int* ring_neighbors);
    template<int maxRingSize, int maxFRNNSize>
    bool getFRNN(float r, int* ring_neighbors, int* neighbors);

    // -------------------------------------------
    // Experimental operations
    __device__ __host__ bool swapVertices(int v0_index, int v1_index);
};


// -------------------------------------------
// Triangulation Methods
// -------------------------------------------

// -------------------------------------------
// Triangulation constructors

DeviceTriangulation::DeviceTriangulation(HostTriangulation* h_triangulation) : h_triangulation(h_triangulation) {
    allocateMemory();
    transferToDevice();
}

DeviceTriangulation::~DeviceTriangulation() {
    freeMemory();
}

// -------------------------------------------
// memory transfer helpers. It transfers the memory from the host to the device and vice versa
bool DeviceTriangulation::transferToHost() { 
    cudaMemcpy(h_triangulation->m_t.data()  , m_t,   m_t_size   * sizeof(Triangle),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_triangulation->m_he.data() , m_he,  m_he_size  * sizeof(HalfEdge),  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_triangulation->m_v.data()  , m_v,   m_v_size   * sizeof(Vertex),    cudaMemcpyDeviceToHost);
    cudaMemcpy(h_triangulation->m_pos.data(), m_pos, m_pos_size * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return true;
}

bool DeviceTriangulation::transferToDevice() { 
    cudaMemcpy(m_t,   h_triangulation->m_t.data(),   m_t_size   * sizeof(Triangle),  cudaMemcpyHostToDevice);
    cudaMemcpy(m_he,  h_triangulation->m_he.data(),  m_he_size  * sizeof(HalfEdge),  cudaMemcpyHostToDevice);
    cudaMemcpy(m_v,   h_triangulation->m_v.data(),   m_v_size   * sizeof(Vertex),    cudaMemcpyHostToDevice);
    cudaMemcpy(m_pos, h_triangulation->m_pos.data(), m_pos_size * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return true;
}

bool DeviceTriangulation::allocateMemory() {
    m_t_size =   h_triangulation->m_t  .size();
    m_he_size =  h_triangulation->m_he .size();
    m_v_size =   h_triangulation->m_v  .size();
    m_pos_size = h_triangulation->m_pos.size();

    cudaMalloc((void**)&m_t,   m_t_size * sizeof(Triangle));
    cudaMalloc((void**)&m_he,  m_he_size * sizeof(HalfEdge));
    cudaMalloc((void**)&m_v,   m_v_size * sizeof(Vertex));
    cudaMalloc((void**)&m_pos, m_pos_size * sizeof(glm::vec2));

    cudaMalloc((void**)&m_helper_t,  m_t_size * sizeof(int));
    cudaMalloc((void**)&m_helper_he, m_he_size * sizeof(int));
    cudaMalloc((void**)&m_helper_v,  m_v_size * sizeof(int));

    cudaMalloc((void**)&m_flag, sizeof(int));

    cudaDeviceSynchronize();
    return true;
}

bool DeviceTriangulation::freeMemory() { 
    m_t_size   = 0;
    m_he_size  = 0;
    m_v_size   = 0;
    m_pos_size = 0;
    
    cudaFree(m_t);
    cudaFree(m_he);
    cudaFree(m_v);
    cudaFree(m_pos);

    cudaFree(m_helper_t);
    cudaFree(m_helper_he);
    cudaFree(m_helper_v);

    cudaFree(m_flag);
    cudaDeviceSynchronize();
    return true;
}


// -------------------------------------------
// incremental construction

int DeviceTriangulation::findContainingTriangleIndexCheckingAll(glm::vec2 p) {
    return -1;
}

int DeviceTriangulation::findContainingTriangleIndexWalking(glm::vec2 p, int t_index) {
    return -1;
}

// -------------------------------------------
// delonizing

struct DelaunayCheckFunctor {
    // i-> is edge number. HalfEdges are 2*i and 2*i+1
    inline __device__ int operator()(int i, int* m_helper_t, HalfEdge* m_he, glm::vec2* m_pos) {
        int v[4];
        int t[2];
        HalfEdge he[2];

        t[0] = m_he[i * 2].t;
        t[1] = m_he[i * 2 ^ 1].t;

        he[0] = m_he[i * 2];
        he[1] = m_he[i * 2 ^ 1];

        v[0] = he[0].v;
        v[1] = he[1].op;
        v[2] = he[1].v;
        v[3] = he[0].op;

        //if (i*2 < 100 && i*2>50)printf("i: %d v[0]: %d v[1]: %d v[2]: %d v[3]: %d incircle: %d \n", i, v[0], v[1], v[2], v[3], inCircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]]));

        //for (int i = 0; i < 4; i++) {
        //    //check convexity of the bicell
        //    if (!orient2d(m_pos[v[i]], m_pos[v[(i + 1) % 4]], m_pos[v[(i + 2) % 4]]))return false;
        //}

        // if incircle and gets both triangles exclusively, then we can flip safely.
        // Still, we want to flip afterwards to decrease thread divergence
        return (t[0] * t[1] > 0) && inCircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]])>0 && (atomicExch(&m_helper_t[t[0]], i) == -1) && (atomicExch(&m_helper_t[t[1]], i) == -1);
    }
};

struct FlipFunctor {
    // i-> is edge number. HalfEdges are 2*i and 2*i+1
    inline __device__ int operator()(int i, Triangle* m_t, HalfEdge* m_he, Vertex* m_v) {
        f2to2(m_t, m_he, m_v, i*2);
    }
};

bool DeviceTriangulation::delonize() {

    int* flips_done = new int[1];
    flips_done[0] = 0;

    dim3 dimBlock(blocksize);
    dim3 dimGrid((m_he_size / 2 + blocksize - 1) / dimBlock.x);

    DelaunayCheckFunctor dcf;
    FlipFunctor ff;
    thrust::device_ptr<int> dev_helper_he(m_helper_he);

    //printf("Total number of edges: %d\n", m_he_size / 2);

    do {
        cudaMemset(m_helper_he, 0, m_he_size * sizeof(int));
        cudaMemset(m_helper_t, -1, m_t_size * sizeof(int));
        cudaMemset(m_flag, 0, sizeof(int));
        cudaDeviceSynchronize();

        resCollector<DelaunayCheckFunctor> << <dimGrid, dimBlock >> > (m_he_size / 2, m_helper_he, dcf, m_helper_t, m_he, m_pos);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(dev_helper_he, dev_helper_he + m_he_size / 2, dev_helper_he);
        cudaDeviceSynchronize();

        compactIncScanned << <dimGrid, dimBlock >> > (m_he_size / 2, m_helper_he, m_helper_he + m_he_size / 2, m_flag);
        cudaDeviceSynchronize();

        doGivenCompacted<FlipFunctor> << <dimGrid, dimBlock >> > (m_helper_he + m_he_size / 2, m_flag, ff, m_t, m_he, m_v);
        cudaDeviceSynchronize();

        cudaMemcpy(flips_done, m_flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //printf("Flips done [delonize]: %d\n", flips_done[0]);
    } while (flips_done[0] > 0);

    delete[] flips_done;
    return true;
}


// -------------------------------------------
// Moving points

bool DeviceTriangulation::movePoints(glm::vec2* d) {
    dim3 dimBlock(blocksize);
    dim3 dimGrid((m_pos_size + blocksize - 1) / dimBlock.x);
    move_points_kernel<<<dimGrid, dimBlock>>>(m_pos_size,m_pos,d);
    cudaDeviceSynchronize();
    return true;
}
bool DeviceTriangulation::moveFlipflop(int p_index, glm::vec2 d) { return false; }
bool DeviceTriangulation::moveFlipflopDelaunay(int p_index, glm::vec2 d) { return false; }
bool DeviceTriangulation::moveUntangling(int p_index, glm::vec2 d) {
    return false;
}
bool DeviceTriangulation::moveUntanglingDelaunay(int p_index, glm::vec2 d) { return false; }

// -------------------------------------------
// Untangling Mesh

// This is simple in the sense that it only deals with simple untanglings. 
// Not with far away ones.
struct SimpleUntangleCheckFunctor {
    // i-> is edge number. HalfEdges are 2*i and 2*i+1
    inline __device__ int operator()(int i, int* m_helper_t, Triangle* m_t, HalfEdge* m_he, Vertex* m_v, glm::vec2* m_pos) {
        int he_index = i*2;
        if (isCreased(m_t, m_he, m_pos, he_index)) {
            int he_upright = he_index;
            int he_inverted = he_index^1;

            // we assume that he_upright is the upright one, and he_inverted is the inverted one
            if (!isCCW(m_t, m_he, m_pos, m_he[he_upright].t))__swap(&he_upright, &he_inverted);

            glm::vec2 op_upright = m_pos[m_he[he_upright].op];
            glm::vec2 op_inverted = m_pos[m_he[he_inverted].op];

            int t_index_upright = m_he[he_upright].t;
            int t_index_inverted = m_he[he_inverted].t;

            // --------------------
            // A-Step
            // Case where either triangle has 0 area
            // TODO: implement this

            // --------------------
            // B-Step
            // Case where the inverted triangle is inside the upright one            
            if (isInside(m_t, m_he, m_pos, op_inverted, t_index_upright)) {
                return (atomicExch(&m_helper_t[t_index_upright], i) == -1) && (atomicExch(&m_helper_t[t_index_inverted], i) == -1);
            }

            // --------------------
            // C-Step
            // Case where the upright triangle is inside the inverted one
            else if (isInsideInverted(m_t, m_he, m_pos, op_upright, t_index_inverted)) {
                return (atomicExch(&m_helper_t[t_index_upright], i) == -1) && (atomicExch(&m_helper_t[t_index_inverted], i) == -1);
            }

            // --------------------
            // D-Step
            // Case where either triangle is inside the other
            // Here there's a possible flip, but you end up with another 2 inverted triangles
            // This can lead to other flips that untangle the triangulation


            // --------------------
            // E-Step
            // Case where either triangle is inside the other, but you end up with another 2 inverted triangles (again)
            // This cannot lead to other flips that untangle the triangulation (that's why this is another case)
        }
        return false;

    }
};

bool DeviceTriangulation::untangle() {
    //m_flag stores whether or not theres something to do, and how much flips can be made in this iteration

    int* flips_done = new int[1];
    flips_done[0] = 0;

    dim3 dimBlock(blocksize);
    dim3 dimGrid((m_he_size / 2 + blocksize - 1) / dimBlock.x);

    SimpleUntangleCheckFunctor sucf;
    FlipFunctor ff;
    thrust::device_ptr<int> dev_helper_he(m_helper_he);

    //printf("Total number of edges: %d\n", m_he_size / 2);

    do {
        cudaMemset(m_helper_he, 0, m_he_size * sizeof(int));
        cudaMemset(m_helper_t, -1, m_t_size * sizeof(int));
        cudaMemset(m_flag, 0, sizeof(int));
        cudaDeviceSynchronize();

        resCollector<SimpleUntangleCheckFunctor> << <dimGrid, dimBlock >> > (m_he_size / 2, m_helper_he, sucf, m_helper_t, m_t, m_he, m_v, m_pos);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(dev_helper_he, dev_helper_he + m_he_size / 2, dev_helper_he);
        cudaDeviceSynchronize();

        compactIncScanned << <dimGrid, dimBlock >> > (m_he_size / 2, m_helper_he, m_helper_he + m_he_size / 2, m_flag);
        cudaDeviceSynchronize();

        doGivenCompacted<FlipFunctor> << <dimGrid, dimBlock >> > (m_helper_he + m_he_size / 2, m_flag, ff, m_t, m_he, m_v);
        cudaDeviceSynchronize();

        cudaMemcpy(flips_done, m_flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //printf("Flips done [untangle]: %d\n", flips_done[0]);
    } while (flips_done[0] > 0);

    delete[] flips_done;
    return true;
}


// -------------------------------------------
// Neighborhood searching
template<int maxRingSize>
bool DeviceTriangulation::oneRing(int* ring_neighbors) {
    dim3 dimBlock(blocksize);
    dim3 dimGrid((m_v_size + blocksize - 1) / dimBlock.x);
    computeOneRingNeighbors_kernel<maxRingSize><<<dimGrid, dimBlock>>>(m_he, m_v, m_v_size, ring_neighbors);
    cudaDeviceSynchronize();
    return true;
}

template<int maxRingSize, int maxFRNNSize>
bool DeviceTriangulation::getFRNN(float r, int* ring_neighbors, int* neighbors) {
    dim3 dimBlock(blocksize);
    dim3 dimGrid((m_v_size + blocksize - 1) / dimBlock.x);
    computeNeighbors_kernel<maxRingSize,maxFRNNSize> <<<dimGrid, dimBlock >>> (m_pos, m_v_size, ring_neighbors, neighbors, pow2(r));
    cudaDeviceSynchronize();
    return true;
}

// -------------------------------------------
// Experimental operations
__device__ __host__ bool DeviceTriangulation::swapVertices(int v0_index, int v1_index) {
    glm::vec2 aux = m_pos[v0_index];
    m_pos[v0_index] = m_pos[v1_index];
    m_pos[v1_index] = aux;

    return true;
}

