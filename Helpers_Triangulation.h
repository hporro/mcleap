#pragma once

#include "Common_Triangulation.h"
#include <glm/glm.hpp>

__device__ __host__ inline bool operator==( const glm::vec2& a, const glm::vec2& b);
__device__ __host__ inline bool orient2d(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c);
template<class T>
__device__ __host__ inline T pow2(const T& a);
template<class T>
__device__ __host__ inline void __swap(T*& a, T*& b);
__device__ __host__ inline bool inCircle(  const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d);
__device__ __host__ inline float sdSegment(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p);


// -------------------------------------------
// Common helper functions
__device__ __host__ bool isInside(        const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const glm::vec2& p, const int t_index); // checks wether a point is inside a triangle
__device__ __host__ bool isInsideInverted(const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const glm::vec2& p, const int t_index); // checks wether a point is inside an inverted triangle
__device__ __host__ bool isCCW(           const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const int t_index); // checks wether a triangle is inverted
__device__ __host__ bool isCreased(       const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const int he_index); // checks wether a edge is creased (one triangle upright and one inverted)
__device__ __host__ bool isInvertedEdge(  const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const int he_index);


// -------------------------------------------
// functions
// -------------------------------------------

__device__ __host__ inline bool operator==(const glm::vec2& a, const glm::vec2& b) {
    return (abs(a.x - b.x) < EPS && abs(a.y - b.y) < EPS);
}

__device__ __host__ inline bool orient2d(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    //glm::vec2 x = b - a;
    //glm::vec2 y = c - a;
    double l  = ((double)b.x - (double)a.x) * ((double)c.y - (double)a.y);
    double r = ((double)b.y - (double)a.y) * ((double)c.x - (double)a.x);

    double det = l - r;
    
    // Shewchuk's dynamic filtering
    //double detsum = 0;
    //
    //if (l > 0.0) {
    //    if (r <= 0.0) { return det > 0; }
    //    else { detsum = l + r; }
    //}
    //else if (l < 0.0) {
    //    if (r >= 0.0) { return det > 0; }
    //    else { detsum = -l - r; }
    //}
    //else { return det; }
    ////return false;
    //double errbound = (3 * UEPS + 16 * UEPS * UEPS) * detsum;
    //if ((det >= errbound) || (-det >= errbound))
    //    return det;

    // Ozaki, K., Bünger, F., Ogita, T., Oishi, S. I., & Rump, S. M. (2016). 
    // Simple floating-point filters for the two-dimensional orientation problem. BIT Numerical Mathematics, 56(2), 729-749.
    //double errbound = theta * (gcem::abs(l + r) + UEPS);
    //if (abs(det) > errbound) {
    //    return det;
    //}

    // This should be replaced with a method of adaptive accuracy or something
    //return false;
    return det > 0;
}

template<class T>
__device__ __host__ inline T pow2(const T& a) {
    return a * a;
}

template<class T>
__device__ __host__ inline void __swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
}

__device__ __host__ inline bool inCircle(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& d) {
    const double a00 = a.x - d.x;
    const double a01 = a.y - d.y;
    const double a02 = pow2(a00) + pow2(a01);
    const double a10 = b.x - d.x;
    const double a11 = b.y - d.y;
    const double a12 = pow2(a10) + pow2(a11);
    const double a20 = c.x - d.x;
    const double a21 = c.y - d.y;
    const double a22 = pow2(a20) + pow2(a21);

    return (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)) > EPS;
}

__device__ __host__ inline float sdSegment(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p) {
    glm::vec2 pa = p - a, ba = b - a;
    float h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h);
}


// -------------------------------------------
// helper functions

__device__ __host__ bool isInside(const Triangle* m_t, const HalfEdge* m_he, const  glm::vec2* m_pos, const glm::vec2& p, const  int t_index) {
    Triangle t = m_t[t_index];
    int v[3];

    v[0] = m_he[t.he].v;
    v[1] = m_he[t.he^1].v;
    v[2] = m_he[t.he].op;

#pragma unroll(3)
    for (int i = 0; i < 3; i++) {
        if (!orient2d(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p))return false;
    }
    return true;
}

__device__ __host__ bool isInsideInverted(const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const glm::vec2& p, const int t_index) {
    Triangle t = m_t[t_index];
    int v[3];

    v[0] = m_he[t.he].v;
    v[1] = m_he[t.he ^ 1].v;
    v[2] = m_he[t.he].op;

#pragma unroll(3)
    for (int i = 0; i < 3; i++) {
        if (orient2d(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p))return false;
    }
    return true;
}

__device__ __host__ bool isCCW(const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const int t_index) {
    Triangle t = m_t[t_index];
    int v[3];

    v[0] = m_he[t.he].v;
    v[1] = m_he[t.he ^ 1].v;
    v[2] = m_he[t.he].op;

    return orient2d(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]]);
}

__device__ __host__ bool isCreased(const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const int he_index) {
    if (m_he[he_index].t == -1)return false;
    if (m_he[he_index ^ 1].t == -1)return false;

    return isCCW(m_t, m_he, m_pos, m_he[he_index ^ 1].t) != isCCW(m_t, m_he, m_pos, m_he[he_index].t);
}

__device__ __host__ bool isInvertedEdge(const Triangle* m_t, const HalfEdge* m_he, const glm::vec2* m_pos, const int he_index) {
    if (m_he[he_index].t == -1)return false;
    if (m_he[he_index ^ 1].t == -1)return false;

    return (!isCCW(m_t, m_he, m_pos, m_he[he_index ^ 1].t)) && (!isCCW(m_t, m_he, m_pos, m_he[he_index].t));
}
