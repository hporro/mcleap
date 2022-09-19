#pragma once

#include "Common_Triangulation.h"
#include <glm/glm.hpp>

__device__ __host__ inline float cross(glm::vec2 a, glm::vec2 b);
__device__ __host__ inline bool operator==(glm::vec2& a, glm::vec2& b);
__device__ __host__ inline bool isToTheLeft(glm::vec2& a, glm::vec2& b, glm::vec2& c);
template<class T>
__device__ __host__ inline T pow2(T& a);
__device__ __host__ inline bool inCircle(glm::vec2& a, glm::vec2& b, glm::vec2& c, glm::vec2& d);
__device__ __host__ inline float sdSegment(glm::vec2 a, glm::vec2 b, glm::vec2 p);


// -------------------------------------------
// Common helper functions
__device__ __host__ bool isInside(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2 p, int t_index); // checks wether a point is inside a triangle
__device__ __host__ bool isInsideInverted(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2 p, int t_index); // checks wether a point is inside an inverted triangle
__device__ __host__ bool isCCW(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int t_index); // checks wether a triangle is inverted
__device__ __host__ bool isCreased(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int he_index); // checks wether a edge is creased (one triangle upright and one inverted)
__device__ __host__ bool isInvertedEdge(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int he_index);


// -------------------------------------------
// functions
// -------------------------------------------

__device__ __host__ inline float cross(glm::vec2 a, glm::vec2 b) {
    return (a.x * b.y) - (a.y * b.x);
}

__device__ __host__ inline bool operator==(glm::vec2& a, glm::vec2& b) {
    return (abs(a.x - b.x) < EPS && abs(a.y - b.y) < EPS);
}

__device__ __host__ inline bool isToTheLeft(glm::vec2& a, glm::vec2& b, glm::vec2& c) {
    return cross(b - a, c - a) > EPS;
}

template<class T>
__device__ __host__ inline T pow2(T& a) {
    return a * a;
}

__device__ __host__ inline bool inCircle(glm::vec2& a, glm::vec2& b, glm::vec2& c, glm::vec2& d) {
    float a00 = a.x - d.x;
    float a01 = a.y - d.y;
    float a02 = pow2(a00) + pow2(a01);
    float a10 = b.x - d.x;
    float a11 = b.y - d.y;
    float a12 = pow2(a10) + pow2(a11);
    float a20 = c.x - d.x;
    float a21 = c.y - d.y;
    float a22 = pow2(a20) + pow2(a21);

    return (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)) > EPS;
}

__device__ __host__ inline float sdSegment(glm::vec2 a, glm::vec2 b, glm::vec2 p) {
    glm::vec2 pa = p - a, ba = b - a;
    float h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h);
}


// -------------------------------------------
// helper functions

__device__ __host__ bool isInside(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2 p, int t_index) {
    Triangle t = m_t[t_index];
    int v[3];
    HalfEdge he[3];
    he[0] = m_he[t.he];
    he[1] = m_he[he[0].next];
    he[2] = m_he[he[1].next];
    v[0] = he[0].v;
    v[1] = he[1].v;
    v[2] = he[2].v;

    for (int i = 0; i < 3; i++) {
        if (!isToTheLeft(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p))return false;
    }
    return true;
}

__device__ __host__ bool isInsideInverted(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, glm::vec2 p, int t_index) {
    Triangle t = m_t[t_index];
    int v[3];
    HalfEdge he[3];
    he[0] = m_he[t.he];
    he[1] = m_he[he[0].next];
    he[2] = m_he[he[1].next];
    v[0] = he[0].v;
    v[1] = he[1].v;
    v[2] = he[2].v;

    for (int i = 0; i < 3; i++) {
        if (isToTheLeft(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p))return false;
    }
    return true;
}

__device__ __host__ bool isCCW(Triangle* m_t, HalfEdge* m_he, glm::vec2* m_pos, int t_index) {
    int v[3];
    HalfEdge he[3];
    Triangle t = m_t[t_index];
    he[0] = m_he[t.he];
    he[1] = m_he[he[0].next];
    he[2] = m_he[he[1].next];
    v[0] = he[0].v;
    v[1] = he[1].v;
    v[2] = he[2].v;

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
