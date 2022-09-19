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

struct HostTriangulation;


// this implementation does not handle removing defragmentation of the std::vectors
struct HostTriangulation {
    // -------------------------------------------
    // host data
    std::vector<glm::vec2> m_pos;
    std::vector<Vertex> m_v;
    std::vector<HalfEdge> m_he;
    std::vector<Triangle> m_t;
    
    // -------------------------------------------
    // constructors
    HostTriangulation();

    // -------------------------------------------
    // incremental construction
    __device__ __host__ int findContainingTriangleIndexCheckingAll(glm::vec2 p);
    __device__ __host__ int findContainingTriangleIndexWalking(glm::vec2 p, int t_index);
    bool addPoint(glm::vec2 p); // inserts with a 3to1 flip a point in the respective triangle
    bool addDelaunayPoint(glm::vec2 p); // does addPoint, and then delonizes the part of the triangulation that breaks the delaunay condition inserting said point

    // -------------------------------------------
    // Add many points to the triangulation
    // adding many points has the advantage that the triangle used to guess where to add next, is likely to be close if the points are sorted
    // we use as a guess the last triangle where we added a point
    bool sortPointsToAdd(std::vector<glm::vec2> points);
    bool addPoints(std::vector<glm::vec2> points); // inserts with a 3to1 flip many points in the respective triangle
    bool addDelaunayPoints(std::vector<glm::vec2> points); // addDelaunayPoint for many points.

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

HostTriangulation::HostTriangulation() {

    m_pos.push_back(glm::vec2(cood_min, cood_min));
    m_pos.push_back(glm::vec2(cood_max, cood_min));
    m_pos.push_back(glm::vec2(cood_max, cood_max));
    m_pos.push_back(glm::vec2(cood_min, cood_max));

    m_v.push_back(Vertex{ 0,0 });
    m_v.push_back(Vertex{ 1,2 });
    m_v.push_back(Vertex{ 2,6 });
    m_v.push_back(Vertex{ 3,8 });

    m_he.push_back(HalfEdge{ 2,0,0,-1 });
    m_he.push_back(HalfEdge{ 9,1,-1,-1 });

    m_he.push_back(HalfEdge{ 4,1,0,1 });
    m_he.push_back(HalfEdge{ 1,2,-1,-1 });

    m_he.push_back(HalfEdge{ 0,2,0,-1 });
    m_he.push_back(HalfEdge{ 6,0,1,-1 });

    m_he.push_back(HalfEdge{ 8,2,1,-1 });
    m_he.push_back(HalfEdge{ 3,3,-1,-1 });

    m_he.push_back(HalfEdge{ 5,3,1,-1 });
    m_he.push_back(HalfEdge{ 7,0,-1,-1 });

    m_t.push_back({ 4 });
    m_t.push_back({ 5 });
}

// -------------------------------------------
// incremental construction

int HostTriangulation::findContainingTriangleIndexCheckingAll(glm::vec2 p) {
    for (int t = 0; t < m_t.size(); t++) {
        if (isInside(m_t.data(), m_he.data(), m_pos.data(), p, t))return t;
    }
    return -1;
}

int HostTriangulation::findContainingTriangleIndexWalking(glm::vec2 p, int t_index) {
    if (isInside(m_t.data(), m_he.data(), m_pos.data(), p, t_index))return t_index;
    
    int he[3];
    he[0] = m_t[t_index].he;
    he[1] = m_he[he[0]].next;
    he[2] = m_he[he[1]].next;

    glm::vec2 p0 = m_pos[m_he[he[0]].v];
    glm::vec2 p1 = m_pos[m_he[he[1]].v];
    glm::vec2 p2 = m_pos[m_he[he[2]].v];

    glm::vec2 centroid = (p0+p1+p2)/3.f;

    for (int i = 0; i < 3; i++) {
        glm::vec2 s = m_pos[m_he[he[i]].v];
        glm::vec2 r = m_pos[m_he[m_he[he[i]].next].v];

        //checking if s-r intersects centroid-p
        if((isToTheLeft(p,centroid,s) != isToTheLeft(p,centroid,r)) && (isToTheLeft(s,r,centroid) != isToTheLeft(s,r,p))) {
            return findContainingTriangleIndexWalking(p, m_he[he[i]].t);
        }
    }

    return -1;
}

bool HostTriangulation::addPoint(glm::vec2 p) {
    int t_index = findContainingTriangleIndexCheckingAll(p);
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

bool HostTriangulation::addDelaunayPoint(glm::vec2 p) {
    int t_index = findContainingTriangleIndexCheckingAll(p);
    if (t_index == -1)return false; // couldnt find the triangle for some reason
    m_pos.push_back(p);
    m_v.push_back(Vertex{ (int)m_pos.size() - 1 });

    // allocate new memory
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
    delonizeTriangle(t_index);
    delonizeTriangle(m_t.size() - 1);
    delonizeTriangle(m_t.size() - 2);
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


//returns true if theres some flipping
bool HostTriangulation::delonizeEdge(int he_index) {

    if (m_he[he_index].t == -1)return false;
    if (m_he[he_index ^ 1].t == -1)return false;

    int v[4];
    HalfEdge he[4];
    he[0] = m_he[he_index];
    he[1] = m_he[he[0].next];
    he[2] = m_he[he[1].next];
    he[3] = m_he[he_index ^ 1];
    v[0] = he[0].v;
    v[1] = m_he[m_he[m_he[he_index ^ 1].next].next].v;
    v[2] = he[1].v;
    v[3] = he[2].v;
    for (int i = 0; i < 4; i++) {
        //check convexity of the bicell
        if (!isToTheLeft(m_pos[v[i]], m_pos[v[(i + 1) % 4]], m_pos[v[(i + 2) % 4]]))return false;
    }
    if (inCircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]])) {
        f2to2(m_t.data(), m_he.data(), m_v.data(),he_index);
        int ihe[4];
        ihe[0] = m_he[he_index].next;
        ihe[1] = m_he[ihe[0]].next;
        ihe[2] = m_he[he_index ^ 1].next;
        ihe[3] = m_he[ihe[2]].next;
        for (int i = 0; i < 4; i++) {
            delonizeEdge(ihe[i]);
        }
        return true;
    }
    return false;
}

//returns true if there was some flipping
bool HostTriangulation::delonizeTriangle(int t_index) {
    Triangle t = m_t[t_index];
    int v[3];
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
    int initial_he = m_v[v_index].he;

    if (m_he[initial_he].t == -1)initial_he = initial_he^1;
    if (delonizeTriangle(m_he[initial_he].t)) { delonizeVertex(v_index); return true; }

    int curr_he = NEXT_OUTGOING(initial_he);
    //printf("\n\n");
    //printf("init_he: %d\n", initial_he);

    while (curr_he != initial_he) {
        //printf("curr_he: %d\n", curr_he);
        if (m_he[curr_he].t == -1)break;
        if (delonizeTriangle(m_he[curr_he].t)) {
            delonizeVertex(v_index);
            return true;
        }
        curr_he = NEXT_OUTGOING(curr_he);
    }
    // is necessary to go backwards too, in the cases close to the border of the triangulation (connected to the 4 vertices enclosing it)
    curr_he = PREV_OUTGOING(initial_he);
    //printf("backwards\n");
    while (curr_he != initial_he) {
        //printf("curr_he: %d\n", curr_he);
        if (m_he[curr_he].t == -1)break;
        if (delonizeTriangle(m_he[curr_he].t)) {
            delonizeVertex(v_index);
            return true;
        }
        curr_he = PREV_OUTGOING(curr_he);
    }
    return false;
}

#undef NETX_OUTGOING
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
            if (isInside(m_t.data(), m_he.data(), m_pos.data(), op_inverted, t_index_upright)) {
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