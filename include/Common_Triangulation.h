#pragma once

#ifndef __NVCC__
#define __global__
#define __device__
#define __host__
#endif

#include <gcem.hpp>

// min and max determine the coordinates of the 4 points enclosing the triangulation
constexpr float cood_min = -10000000;
constexpr float cood_max = 10000000;
constexpr float EPS = 0.00000000000000001f;
constexpr double UEPS = std::numeric_limits<double>::epsilon();
constexpr double phi = 2.0 * gcem::floor((-1 + gcem::sqrt(4.0 * 1.0 / UEPS + 45.0)) / 4.0);
constexpr double theta = 3.0 * UEPS - (phi - 22.0) * UEPS * UEPS;
constexpr double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111;

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
__device__ __host__ inline void       f2to2(Triangle* m_t,HalfEdge* m_he, Vertex* m_v, const int he_index);
__device__ __host__ inline void       f1to3(Triangle* m_t,HalfEdge* m_he, Vertex* m_v, const f3to1_info& finfo, const int t_index);
__device__ __host__ inline void       f2to4(Triangle* m_t,HalfEdge* m_he, Vertex* m_v, const int v_index, const int he_index);
__device__ __host__ inline f3to1_info f3to1(Triangle* m_t,HalfEdge* m_he, Vertex* m_v, const int v_index, const int t_index);


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

__device__ __host__ inline void f2to2(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, const int he_index) {
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

    m_he[he[0]].op = v[1];
    m_he[he[5]].op = v[2];
    m_he[he[2]].op = v[3];

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

    m_he[he[1]].op = v[0];
    m_he[he[3]].op = v[3];
    m_he[he[4]].op = v[2];
}

__device__ __host__ inline void f1to3(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, const f3to1_info& finfo, const int t_index) {
    int he[9];
    int t[3];
    int v[4];

    t[0] = t_index;
    t[1] = finfo.t0_index;
    t[2] = finfo.t1_index;

    he[0] = m_t[t_index].he;
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
    v[3] = finfo.v_index;

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

    m_he[he[0]].op = v[3]; //0
    m_he[he[3]].op = v[0]; //1
    m_he[he[5]].op = v[1]; //3

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

    m_he[he[1]].op = v[3];
    m_he[he[7]].op = v[1];
    m_he[he[4]].op = v[2];

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

    m_he[he[2]].op = v[3];
    m_he[he[6]].op = v[2];
    m_he[he[8]].op = v[0];
}

__device__ __host__ inline void f2to4(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, const int v_index, const int he_index) {}
__device__ __host__ inline f3to1_info f3to1(Triangle* m_t, HalfEdge* m_he, Vertex* m_v, const int v_index, const int t_index) {
    return {}; 
}

