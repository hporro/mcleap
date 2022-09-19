#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <stack>

#include <glm/glm.hpp>
#include <glm/vec2.hpp>

#include "Common_Triangulation.h"
#include "Helpers_Triangulation.h"

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
    size_t m_pos_size;
    size_t m_v_size;
    size_t m_he_size;
    size_t m_t_size;

    glm::vec2* m_pos;
    Vertex* m_v;
    HalfEdge* m_he;
    Triangle* m_t;
    // helper device data


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
    bool movePoints(int p_index, glm::vec2 d); //moves without fixing anything
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
    bool oneRing(int v_index);
    bool getFRNN(int v_index, float r);

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
// memory transfer helpers. It transfers the memory from the host to the device or vice versa
bool DeviceTriangulation::transferToHost() { 
    cudaMemcpy(h_triangulation->m_t.data()  , m_t, m_t_size * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(h_triangulation->m_he.data() , m_he, m_t_size * sizeof(HalfEdge), cudaMemcpyHostToDevice);
    cudaMemcpy(h_triangulation->m_v.data()  , m_v, m_t_size * sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(h_triangulation->m_pos.data(), m_pos, m_t_size * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    return true;
}
bool DeviceTriangulation::transferToDevice() { 
    cudaMemcpy(m_t,   h_triangulation->m_t.data(),   m_t_size * sizeof(Triangle),  cudaMemcpyHostToDevice);
    cudaMemcpy(m_he,  h_triangulation->m_he.data(),  m_t_size * sizeof(HalfEdge),  cudaMemcpyHostToDevice);
    cudaMemcpy(m_v,   h_triangulation->m_v.data(),   m_t_size * sizeof(Vertex),    cudaMemcpyHostToDevice);
    cudaMemcpy(m_pos, h_triangulation->m_pos.data(), m_t_size * sizeof(glm::vec2), cudaMemcpyHostToDevice);

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

bool DeviceTriangulation::delonize() {
    return false;
}

// -------------------------------------------
// Moving points

bool DeviceTriangulation::moveFlipflop(int p_index, glm::vec2 d) { return false; }
bool DeviceTriangulation::moveFlipflopDelaunay(int p_index, glm::vec2 d) { return false; }
bool DeviceTriangulation::moveUntangling(int p_index, glm::vec2 d) {
    return false;
}
bool DeviceTriangulation::moveUntanglingDelaunay(int p_index, glm::vec2 d) { return false; }

// -------------------------------------------
// Untangling Mesh
bool DeviceTriangulation::untangle() {
    
    return false;
}


// -------------------------------------------
// Neighborhood searching
bool DeviceTriangulation::oneRing(int v_index) {
    return false;
}


bool DeviceTriangulation::getFRNN(int v_index, float r) {
    return false;
}

// -------------------------------------------
// Experimental operations
__device__ __host__ bool DeviceTriangulation::swapVertices(int v0_index, int v1_index) {
    glm::vec2 aux = m_pos[v0_index];
    m_pos[v0_index] = m_pos[v1_index];
    m_pos[v1_index] = aux;

    return true;
}

