#pragma once

#ifndef __NVCC__
#define __global__
#define __device__
#define __host__
#endif

// min and max determine the coordinates of the 4 points enclosing the triangulation
constexpr float cood_min = -10000000;
constexpr float cood_max = 10000000;
constexpr float EPS = 0.00000000000000001f;

// -------------------------------------------
// Structs defined
// -------------------------------------------

struct Vertex;
struct Edge;
struct HalfEdge;
struct Triangle;
struct f3to1_info;

// -------------------------------------------
// basic flips
__device__ __host__ inline bool       f2to2(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int he_index);
__device__ __host__ inline bool       f1to3(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, f3to1_info finfo, int t_index);
__device__ __host__ inline bool       f2to4(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int v_index, int he_index);
__device__ __host__ inline f3to1_info f3to1(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int v_index, int t_index);


// -------------------------------------------
// structs
// -------------------------------------------

struct Vertex {
    int pos_index; // this is not used still, but the idea is to use it to implement vertex swaping
    int he; // halfedge index that goes from this vertex
};

struct HalfEdge {
    // if a halfedge has an index i in the triangulation, then i^1 is its twin, so its not necessary to store that index.
    int next; // next halfedge
    int v; // this half edge comes from v
    int t; // triangle that contains t
    int op; // opposite vertex (not in effective use yet)
};

struct Triangle {
    int he; // half edge inside this triangle
};

struct f3to1_info {
    int v_index;
    int t0_index, t1_index;
    int he0_index, he2_index, he4_index;
};


// -------------------------------------------
// flipping

__device__ __host__ inline bool f2to2(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int he_index) {
    int v[4];
    int he[6];
    int t[2];
    he[0] = he_index;
    he[1] = he_index ^ 1;
    he[2] = m_he[he[0]].next;
    he[3] = m_he[he[2]].next;
    he[4] = m_he[he[1]].next;
    he[5] = m_he[he[4]].next;

    t[0] = m_he[he[0]].t;
    t[1] = m_he[he[1]].t;

    v[0] = m_he[he[0]].v;
    v[1] = m_he[he[1]].v;
    v[2] = m_he[he[3]].v;
    v[3] = m_he[he[5]].v;

    m_t[t[0]].he = he[0];
    m_t[t[1]].he = he[1];

    m_v[v[1]].he = he[2];
    m_v[v[0]].he = he[4];

    // t0
    m_he[he[0]].next = he[5];
    m_he[he[5]].next = he[2];
    m_he[he[2]].next = he[0];

    m_he[he[0]].v = v[2];
    m_he[he[5]].v = v[3];
    m_he[he[2]].v = v[1];

    m_he[he[0]].t = t[0];
    m_he[he[5]].t = t[0];
    m_he[he[2]].t = t[0];

    // t1
    m_he[he[1]].next = he[3];
    m_he[he[3]].next = he[4];
    m_he[he[4]].next = he[1];

    m_he[he[1]].v = v[3];
    m_he[he[3]].v = v[2];
    m_he[he[4]].v = v[0];

    m_he[he[1]].t = t[1];
    m_he[he[3]].t = t[1];
    m_he[he[4]].t = t[1];

    return true;
}

__device__ __host__ inline bool f1to3(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, f3to1_info finfo, int t_index) {
    int he[9];
    int t[3];
    int v[4];

    int v_index = finfo.v_index;

    Triangle tr = m_t[t_index];

    t[0] = t_index;
    t[1] = finfo.t0_index;
    t[2] = finfo.t1_index;

    he[0] = tr.he;
    he[1] = m_he[he[0]].next;
    he[2] = m_he[he[1]].next;
    he[8] = finfo.he0_index;
    he[7] = finfo.he0_index^1;
    he[6] = finfo.he2_index;
    he[5] = finfo.he2_index^1;
    he[4] = finfo.he4_index;
    he[3] = finfo.he4_index^1;

    v[0] = m_he[he[0]].v;
    v[1] = m_he[he[1]].v;
    v[2] = m_he[he[2]].v;
    v[3] = v_index;

    m_t[t[0]].he = he[0];
    m_t[t[1]].he = he[1];
    m_t[t[2]].he = he[2];

    m_v[v[3]].he = he[4];

    // t0
    m_he[he[0]].next = he[3];
    m_he[he[3]].next = he[5];
    m_he[he[5]].next = he[0];

    m_he[he[0]].v = v[0];
    m_he[he[3]].v = v[1];
    m_he[he[5]].v = v[3];

    m_he[he[0]].t = t[0];
    m_he[he[3]].t = t[0];
    m_he[he[5]].t = t[0];

    // t1
    m_he[he[1]].next = he[7];
    m_he[he[7]].next = he[4];
    m_he[he[4]].next = he[1];

    m_he[he[1]].v = v[1];
    m_he[he[7]].v = v[2];
    m_he[he[4]].v = v[3];

    m_he[he[1]].t = t[1];
    m_he[he[7]].t = t[1];
    m_he[he[4]].t = t[1];

    // t2
    m_he[he[2]].next = he[6];
    m_he[he[6]].next = he[8];
    m_he[he[8]].next = he[2];

    m_he[he[2]].v = v[2];
    m_he[he[6]].v = v[0];
    m_he[he[8]].v = v[3];

    m_he[he[2]].t = t[2];
    m_he[he[6]].t = t[2];
    m_he[he[8]].t = t[2];

    return true;
}

__device__ __host__ inline bool f2to4(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int v_index, int he_index) { return false; }
__device__ __host__ inline f3to1_info f3to1(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int v_index, int t_index) { return {}; }

