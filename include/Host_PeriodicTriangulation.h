#pragma once

//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
//#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>

#include <fstream>
#include <cassert>
#include <list>
#include <vector>
#include <stack>
#include <queue>

//typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef CGAL::Periodic_2_Delaunay_triangulation_traits_2<K> GT;
//typedef CGAL::Periodic_2_Delaunay_triangulation_2<GT>       PDT;
//typedef PDT::Face_handle                                    Face_handle;
//typedef PDT::Vertex_handle                                  Vertex_handle;
//typedef PDT::Locate_type                                    Locate_type;
//typedef PDT::Point                                          Point;
//typedef PDT::Iso_rectangle                                  Iso_rectangle;

#include <glm/glm.hpp>
#include <glm/vec2.hpp>

#include <libmorton/morton.h>

#include "Common_Triangulation.h"
#include "Helpers_Triangulation.h"


// -------------------------------------------
// Structs defined
// -------------------------------------------

struct HostTriangulation;


// this implementation does not handle removing defragmentation of the std::vectors
struct HostTriangulation {
    // -------------------------------------------
    // host data
    //PDT T;
    std::vector<glm::vec2> m_pos;
    std::vector<Vertex> m_v;
    std::vector<HalfEdge> m_he;
    std::vector<Triangle> m_t;

    // -------------------------------------------
    // constructors
    HostTriangulation(double x1, double y1, double x2, double y2);

    // -------------------------------------------
    // incremental construction
    bool addPoint(glm::vec2 p, int t_index); // inserts with a 3to1 flip a point in the respective triangle
    bool addPoint(glm::vec2 p); // inserts with a 3to1 flip a point in the respective triangle

    bool addDelaunayPoint(glm::vec2 p); // does addPoint, and then delonizes the part of the triangulation that breaks the delaunay condition inserting said point

    // -------------------------------------------
    // Add many points to the triangulation
    // adding many points has the advantage that the triangle used to guess where to add next, is likely to be close if the points are sorted
    // we use as a guess the last triangle where we added a point
    bool sortPointsToAdd(std::vector<glm::vec2> points);
    bool addPoints(std::vector<glm::vec2> points); // inserts with a 3to1 flip many points in the respective triangle
    bool addDelaunayPoints(std::vector<glm::vec2>& points); // addDelaunayPoint for many points.

    bool isInside(glm::vec2& p, int t_index) {
        Triangle t = m_t[t_index];
        int v[3];

        v[0] = m_he[t.he].v;
        v[1] = m_he[t.he ^ 1].v;
        v[2] = m_he[t.he].op;

        for (int i = 0; i < 3; i++) {
            if (orient2d(m_pos[v[i]], m_pos[v[(i + 1) % 3]], p) <= 0)return false;
        }
        return true;
    }

    // -------------------------------------------
    // delonizing
    bool delonize();
    bool delonizeEdge(int he_index);
    bool delonizeTriangle(int t_index);
    bool delonizeVertex(int v_index);

    // -------------------------------------------
    // Moving points
    bool movePoint(int p_index, glm::vec2 d); //moves without fixing anything
    bool moveFlipflop(int p_index, glm::vec2 d); //removes and reinserts a point
    bool moveFlipflopDelaunay(int p_index, glm::vec2 d); //removes and reinserts a point, and then Delonizates
    bool moveUntangling(int p_index, glm::vec2 d); //moves without fixing anything, and then untangles
    bool moveUntanglingDelaunay(int p_index, glm::vec2 d); //moves without fixing anything, and then untangles, and then Delonizates

    // -------------------------------------------
    // Untangling Mesh
    // based on Shewchuk's untangling
    bool untangle();
    bool untangleEdge(int he_index);
    bool untangleTriangle(int he_index);

    // -------------------------------------------
    // Neighborhood searching
    std::vector<int> oneRing(int v_index);
    std::vector<int> getFRNN(int v_index, float r);

    // -------------------------------------------
    // Experimental operations
    __device__ __host__ bool swapVertices(int v0_index, int v1_index);

};


// -------------------------------------------
// Triangulation Methods
// -------------------------------------------

// -------------------------------------------
// Triangulation constructors
HostTriangulation::HostTriangulation(double x1, double y1, double x2, double y2) {
    //Iso_rectangle domain(x1, y1, x2, y2);
    //T(domain);
}

// -------------------------------------------
// incremental construction

bool HostTriangulation::addPoint(glm::vec2 p, int t_index) {
    if (t_index == -1)return false; // couldnt find the triangle for some reason
    m_pos.push_back(p);
    m_v.push_back(Vertex{ (int)m_pos.size() - 1 });

    //allocate new memory
    m_t.push_back(Triangle{});
    m_t.push_back(Triangle{});

    m_he.push_back(HalfEdge{});
    m_he.push_back(HalfEdge{});
    m_he.push_back(HalfEdge{});
    m_he.push_back(HalfEdge{});
    m_he.push_back(HalfEdge{});
    m_he.push_back(HalfEdge{});

    f3to1_info finfo{
        (int)m_v.size() - 1,
        (int)m_t.size() - 1,
        (int)m_t.size() - 2,
        (int)m_he.size() - 1,
        (int)m_he.size() - 3,
        (int)m_he.size() - 5
    };

    f1to3(m_t.data(), m_he.data(), m_v.data(), finfo, t_index);
    return true;
}

bool HostTriangulation::addPoint(glm::vec2 p) {
    return false;
}

bool HostTriangulation::addDelaunayPoint(glm::vec2 p) {
    return false;
}


// -------------------------------------------
// Add many points to the triangulation
// adding many points has the advantage that the triangle used to guess where to add next, is likely to be close if the points are sorted
// we use as a guess the last triangle where we added a point
struct cmp_points {
    bool operator()(glm::vec2& a, glm::vec2& b) {
        uint_fast32_t am = libmorton::m2D_e_LUT< uint_fast32_t, uint_fast32_t>(a.x, a.y);
        uint_fast32_t bm = libmorton::m2D_e_LUT< uint_fast32_t, uint_fast32_t>(b.x, b.y);
        return am < bm;
        return true;
    }
};

bool HostTriangulation::addDelaunayPoints(std::vector<glm::vec2>& points) {
    std::sort(points.begin(), points.end(), cmp_points{});
    for (auto p : points)addDelaunayPoint(p);
    return true;
}

// -------------------------------------------
// delonizing

bool HostTriangulation::delonize() {
    bool flipped = false;
    bool ready = false;
    while (!ready) {
        ready = true;
        for (int i = 0; i < m_he.size(); i += 2) {
            if (delonizeEdge(i)) {
                ready = false;
                flipped = true;
                break;
            }
        }
    }
    return flipped;
}


//returns true if theres some legalizing done
bool HostTriangulation::delonizeEdge(int he_index) {

    if (m_he[he_index].t == -1)return false;
    if (m_he[he_index ^ 1].t == -1)return false;

    int v[4];

    v[0] = m_he[he_index].v;
    v[1] = m_he[he_index ^ 1].op;
    v[2] = m_he[he_index ^ 1].v;
    v[3] = m_he[he_index].op;

    //for (int i = 0; i < 4; i++) {
    //    //check convexity of the bicell
    //    if (!orient2d(m_pos[v[i]], m_pos[v[(i + 1) % 4]], m_pos[v[(i + 2) % 4]]))return false;
    //}

    if (inCircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]]) > 0) {
        //if (angle_incircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]]) > 1.0000001) {
        f2to2(m_t.data(), m_he.data(), m_v.data(), he_index);
        return true;
    }
    return false;
}

//returns true if there was some flipping
//legalizes the edges around
bool HostTriangulation::delonizeTriangle(int t_index) {
    Triangle t = m_t[t_index];
    int he[3];
    he[0] = t.he;
    he[1] = m_he[he[0]].next;
    he[2] = m_he[he[1]].next;

    bool delonized = false;
    for (int i = 0; i < 3; i++) {
        //OPTIMIZATION: make an iterative version of the delonization thing
        delonized |= delonizeEdge(he[i]);
    }
    return delonized;
}

#define NEXT_OUTGOING(he_index) m_he[m_he[he_index].next].next^1
#define PREV_OUTGOING(he_index) m_he[he_index^1].next

bool HostTriangulation::delonizeVertex(int v_index) {
    int initial_outgoing_he = m_v[v_index].he;

    //if (m_he[initial_outgoing_he].t == -1)initial_outgoing_he = initial_outgoing_he ^ 1;

    int curr_outgoing_he = initial_outgoing_he;

    do {
        int curr_link_he = m_he[curr_outgoing_he].next;
        while (delonizeEdge(curr_link_he)) {
            curr_link_he = m_he[curr_outgoing_he].next;
        }
        curr_outgoing_he = NEXT_OUTGOING(curr_outgoing_he);
    } while (curr_outgoing_he != initial_outgoing_he);

    curr_outgoing_he = initial_outgoing_he;

    //do {
    //    int curr_link_he = m_he[m_he[curr_outgoing_he].next].next;
    //    while (delonizeEdge(curr_link_he)) {
    //        curr_link_he = m_he[m_he[curr_outgoing_he].next].next;
    //    }
    //    curr_outgoing_he = PREV_OUTGOING(curr_outgoing_he);
    //} while (curr_outgoing_he != initial_outgoing_he);


    return false;
}

#undef NEXT_OUTGOING
#undef PREV_OUTGOING

// -------------------------------------------
// Moving points

bool HostTriangulation::movePoint(int p_index, glm::vec2 d) {
    m_pos[p_index] += d;
    return true;
}

bool HostTriangulation::moveFlipflop(int p_index, glm::vec2 d) { return false; }
bool HostTriangulation::moveFlipflopDelaunay(int p_index, glm::vec2 d) { return false; }
bool HostTriangulation::moveUntangling(int p_index, glm::vec2 d) {
    return false;
}
bool HostTriangulation::moveUntanglingDelaunay(int p_index, glm::vec2 d) { return false; }

// -------------------------------------------
// Untangling Mesh
bool HostTriangulation::untangle() {
    bool untangled_something = false;
    for (int i = 0; i < m_he.size(); i += 2) {
        untangled_something |= untangleEdge(i);
    }
    return untangled_something;
}

bool HostTriangulation::untangleEdge(int he_index) {

    if (isCreased(m_t.data(), m_he.data(), m_pos.data(), he_index)) {
        int he_upright = he_index;
        int he_inverted = he_index ^ 1;

        if (!isCCW(m_t.data(), m_he.data(), m_pos.data(), m_he[he_upright].t))std::swap(he_upright, he_inverted);

        glm::vec2 op_upright = m_pos[m_he[m_he[m_he[he_upright].next].next].v];
        glm::vec2 op_inverted = m_pos[m_he[m_he[m_he[he_inverted].next].next].v];

        int t_index_upright = m_he[he_upright].t;
        int t_index_inverted = m_he[he_inverted].t;

        // --------------------
        // A-Step
        // Case where either triangle has 0 area
        // TODO: implement this

        // --------------------
        // B-Step
        // Case where the inverted triangle is inside the upright one            
        if (isInside(op_inverted, t_index_upright)) {
            f2to2(m_t.data(), m_he.data(), m_v.data(), he_index);
            return true;
        }

        // --------------------
        // C-Step
        // Case where the upright triangle is inside the inverted one
        else if (isInsideInverted(m_t.data(), m_he.data(), m_pos.data(), op_upright, t_index_inverted)) {
            f2to2(m_t.data(), m_he.data(), m_v.data(), he_index);
            return true;
        }

        // --------------------
        // D-Step
        // Case where either triangle is inside the other


        // --------------------
        // E-Step
        // Case where either triangle is inside the other (again)

    }
    return false;
}

bool HostTriangulation::untangleTriangle(int he_index) {
    int he[3];
    he[0] = he_index;
    he[1] = m_he[he[0]].next;
    he[2] = m_he[he[1]].next;

    bool untangledSomething = false;

    untangledSomething |= untangleEdge(he[0]);
    untangledSomething |= untangleEdge(he[1]);
    untangledSomething |= untangleEdge(he[2]);
    return untangledSomething;
}

// -------------------------------------------
// Neighborhood searching
#define NEXT_OUTGOING(he_index) m_he[m_he[he_index].next].next^1
#define PREV_OUTGOING(he_index) m_he[he_index^1].next

std::vector<int> HostTriangulation::oneRing(int v_index) {
    std::vector<int> oneRing;
    int initial_he = m_v[v_index].he;

    if (m_he[initial_he].t == -1)initial_he = initial_he ^ 1;

    int curr_he = NEXT_OUTGOING(initial_he);

    int adjecent = m_he[initial_he ^ 1].v;
    oneRing.push_back(adjecent);

    while (curr_he != initial_he) {
        if (m_he[curr_he].t == -1)break;
        adjecent = m_he[curr_he ^ 1].v;
        oneRing.push_back(adjecent);
        curr_he = NEXT_OUTGOING(curr_he);
    }
    // is necessary to go backwards too, in the cases close to the border of the triangulation (connected to the 4 vertices enclosing it)
    curr_he = PREV_OUTGOING(initial_he);
    //printf("backwards\n");
    while (curr_he != initial_he) {
        if (m_he[curr_he].t == -1)break;
        adjecent = m_he[curr_he ^ 1].v;
        oneRing.push_back(adjecent);
        curr_he = PREV_OUTGOING(curr_he);
    }
    return oneRing;
}

#undef NETX_OUTGOING
#undef PREV_OUTGOING


std::vector<int> HostTriangulation::getFRNN(int v_index, float r) {

    std::vector<int> neighbors;
    std::vector<bool> visited;

    visited.resize(m_pos.size(), false);

    std::stack<int> queue;

    glm::vec2 curr_pos = m_pos[v_index];
    float r2 = r * r;

    // Mark the current node as visited and enqueue it
    visited[v_index] = true;
    queue.push(v_index);

    while (!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        int s = queue.top();
        queue.pop();

        for (auto adjecent : oneRing(s)) {
            if (!visited[adjecent]) {
                visited[adjecent] = true;
                if (glm::dot(curr_pos - m_pos[adjecent], curr_pos - m_pos[adjecent]) <= r2) {
                    neighbors.push_back(adjecent);
                    queue.push(adjecent);
                }
            }
        }
    }


    return neighbors;
}

// -------------------------------------------
// Experimental operations
__device__ __host__ bool HostTriangulation::swapVertices(int v0_index, int v1_index) {
    glm::vec2 aux = m_pos[v0_index];
    m_pos[v0_index] = m_pos[v1_index];
    m_pos[v1_index] = aux;

    return true;
}
