#pragma once

#include "Common_Triangulation.h"
#include <glm/glm.hpp>

__device__ __host__ inline bool operator==(glm::vec2& a, glm::vec2& b);
__device__ __host__ inline bool isToTheLeft(glm::vec2& a, glm::vec2& b, glm::vec2& c);
template<class T>
__device__ __host__ inline T pow2(T& a);
template<class T>
__device__ __host__ inline void __swap(T*& a, T*& b);
__device__ __host__ inline bool inCircle(glm::vec2& a, glm::vec2& b, glm::vec2& c, glm::vec2& d);
__device__ __host__ inline float sdSegment(glm::vec2& a, glm::vec2& b, glm::vec2& p);


// -------------------------------------------
// Common helper functions
__device__ __host__ bool isInside(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2& p, int t_index); // checks wether a point is inside a triangle
__device__ __host__ bool isInsideInverted(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2& p, int t_index); // checks wether a point is inside an inverted triangle
__device__ __host__ bool isCCW(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int t_index); // checks wether a triangle is inverted
__device__ __host__ bool isCreased(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int he_index); // checks wether a edge is creased (one triangle upright and one inverted)
__device__ __host__ bool isInvertedEdge(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int he_index);


// -------------------------------------------
// functions
// -------------------------------------------

__device__ __host__ inline bool operator==(glm::vec2& a, glm::vec2& b) {
    return (abs(a.x - b.x) < EPS && abs(a.y - b.y) < EPS);
}

__device__ __host__ inline bool isToTheLeft(glm::vec2& a, glm::vec2& b, glm::vec2& c) {
    //glm::vec2 x = b - a;
    //glm::vec2 y = c - a;
    return ((b.x-a.x) * (c.y-a.y)) - ((b.y-a.y) * (c.x-a.x)) > -EPS;
}

template<class T>
__device__ __host__ inline T pow2(T& a) {
    return a * a;
}

template<class T>
__device__ __host__ inline void __swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
}

// FOR SOME REASON (precision, duh) THIS HAS TO BE DOUBLE PRESICION, DO NOT TOUCH (IM TALKING TO YOU HEINICH)
__device__ __host__ inline bool inCircle(glm::vec2& a, glm::vec2& b, glm::vec2& c, glm::vec2& d) {
    double a00 = a.x - d.x;
    double a01 = a.y - d.y;
    double a02 = pow2(a00) + pow2(a01);
    double a10 = b.x - d.x;
    double a11 = b.y - d.y;
    double a12 = pow2(a10) + pow2(a11);
    double a20 = c.x - d.x;
    double a21 = c.y - d.y;
    double a22 = pow2(a20) + pow2(a21);

    return (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)) > EPS;
}

__device__ __host__ inline float sdSegment(glm::vec2 a, glm::vec2 b, glm::vec2 p) {
    glm::vec2 pa = p - a, ba = b - a;
    float h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h);
}


// -------------------------------------------
// helper functions

__device__ __host__ bool isInside(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2& p, int t_index) {
    Triangle t = m_t[t_index];
    int v[3];

    v[0] = m_he[t.he].v;
    v[1] = m_he[t.he^1].v;
    v[2] = m_he[t.he].op;

#pragma unroll(3)
    for (int i = 0; i < 3; i++) {
        if (!isToTheLeft(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p))return false;
    }
    return true;
}

__device__ __host__ bool isInsideInverted(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2& p, int t_index) {
    Triangle t = m_t[t_index];
    int v[3];

    v[0] = m_he[t.he].v;
    v[1] = m_he[t.he ^ 1].v;
    v[2] = m_he[t.he].op;

#pragma unroll(3)
    for (int i = 0; i < 3; i++) {
        if (isToTheLeft(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p))return false;
    }
    return true;
}

__device__ __host__ bool isCCW(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int t_index) {
    Triangle t = m_t[t_index];
    int v[3];

    v[0] = m_he[t.he].v;
    v[1] = m_he[t.he ^ 1].v;
    v[2] = m_he[t.he].op;

    return isToTheLeft(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]]);
}

__device__ __host__ bool isCreased(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int he_index) {
    if (m_he[he_index].t == -1)return false;
    if (m_he[he_index ^ 1].t == -1)return false;

    return isCCW(m_t, m_he, m_pos, m_he[he_index ^ 1].t) != isCCW(m_t, m_he, m_pos, m_he[he_index].t);
}

__device__ __host__ bool isInvertedEdge(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int he_index) {
    if (m_he[he_index].t == -1)return false;
    if (m_he[he_index ^ 1].t == -1)return false;

    return (!isCCW(m_t, m_he, m_pos, m_he[he_index ^ 1].t)) && (!isCCW(m_t, m_he, m_pos, m_he[he_index].t));
}
