#pragma once

#include "Common_Triangulation.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Functor is supposed to output 0 if not and 1 if yes
template<class PredicateFunctor, typename... Args>
__global__ void resCollector(int n, int* res, PredicateFunctor f, Args... args) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] |= f(i,args...);
	}
}

//flag stores how many threads actually do work
__global__ void compactIncScanned(int n, int* scanned, int* compacted, int* flag) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0) {
		if (scanned[0] == 1)compacted[0] = 0;
		flag[0] = scanned[n - 1];
	}
	if (i < n && i>0) {
		if (scanned[i] > scanned[i - 1]) {
			compacted[scanned[i]-1] = i;
		}
	}
}

template<class MutatingFunctor, typename... Args>
__global__ void doGivenCompacted(int* compacted, int* flag, MutatingFunctor f , Args... args) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < flag[0]) {
		f(compacted[i], args...);
	}
}

__global__ void move_points_kernel(int n, glm::vec2* pos, glm::vec2* d) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && i >= 4) { // to not move 4 bounding points
		pos[i] += d[i];
	}
}

__global__ void integrate_move_points_kernel(int n, glm::vec2* pos, float* dt, glm::vec2* v) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && i >= 4) { // to not move 4 bounding points
		pos[i] += v[i] * dt[i];
	}
}

#define PREV_OUTGOING(he_index) m_he[he_index^1].next
// n_he -> number of full edges
template<int maxRingSize>
__global__ void computeOneRingNeighbors_kernel(const HalfEdge* m_he, const Vertex* m_v, int n_v, int* ring_neighbors) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_v) {
		int counter = 0; // number of neighbors this vertex has

		int initial_outgoing_he = m_v[i].he;

		//if (i == 185) printf("i: %d outgoing_he: %d outgoing_triangle: %d\n", i, initial_outgoing_he, m_he[initial_outgoing_he].t);
		//if (m_he[initial_outgoing_he].t == -1)initial_outgoing_he = initial_outgoing_he ^ 1;

		int curr_outgoing_he = initial_outgoing_he;

		do {
			if (counter+2 > maxRingSize)break; // TODO: do something about this case
			
			ring_neighbors[(counter + 1) * n_v + i] = m_he[curr_outgoing_he^1].v;
			counter++;

			//if (i < 100 && i>80)printf("i: %d num_ring_neighbors: %d curr_he: %d\n", i, counter, curr_outgoing_he);
			curr_outgoing_he = PREV_OUTGOING(curr_outgoing_he);
			//if (i < 100 && i>80)printf("i: %d num_ring_neighbors: %d curr_he: %d\n", i, counter, curr_outgoing_he);
		} while (curr_outgoing_he != initial_outgoing_he);

		__syncthreads();

		ring_neighbors[i] = counter; // first space of the vertex reserved to the total number of neighbors
		//if (i < 100 && i>80)printf("i: %d num_ring_neighbors: %d\n", i, counter);

	}
}

#undef PREV_OUTGOING


#define PREV_OUTGOING(he_index) m_he[he_index^1].next
// n_he -> number of full edges
template<int maxRingSize>
__global__ void computeOneRingNeighbors_kernel2(const HalfEdge* m_he, const Vertex* m_v, int n_v, int* ring_neighbors) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_v) {
		int counter = 0; // number of neighbors this vertex has

		int initial_outgoing_he = m_v[i].he;

		//if (i == 185) printf("i: %d outgoing_he: %d outgoing_triangle: %d\n", i, initial_outgoing_he, m_he[initial_outgoing_he].t);
		//if (m_he[initial_outgoing_he].t == -1)initial_outgoing_he = initial_outgoing_he ^ 1;

		int curr_outgoing_he = initial_outgoing_he;

		do {
			if (counter + 2 > maxRingSize)break; // TODO: do something about this case

			ring_neighbors[(counter + 1) + maxRingSize * i] = m_he[curr_outgoing_he ^ 1].v;
			counter++;

			//if (i < 100 && i>80)printf("i: %d num_ring_neighbors: %d curr_he: %d\n", i, counter, curr_outgoing_he);
			curr_outgoing_he = PREV_OUTGOING(curr_outgoing_he);
			//if (i < 100 && i>80)printf("i: %d num_ring_neighbors: %d curr_he: %d\n", i, counter, curr_outgoing_he);
		} while (curr_outgoing_he != initial_outgoing_he);

		__syncthreads();

		ring_neighbors[i* maxRingSize] = counter; // first space of the vertex reserved to the total number of neighbors
		//if (i < 100 && i>80)printf("i: %d num_ring_neighbors: %d\n", i, counter);

	}
}

#undef PREV_OUTGOING

__device__ __host__ float sqrtDist(const glm::vec2& a, const glm::vec2& b) {
	return pow2(a.x-b.x) + pow2(a.y-b.y);
}

template<int maxRingSize>
__global__ void compute_closestNeighbors_kernel(const glm::vec2* m_pos, int n_v, const int* ring_neighbors, int* closest_neighbors) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_v) {
		glm::vec2 i_pos = m_pos[i];
		float closest_dist = sqrtDist(i_pos, m_pos[ring_neighbors[n_v + i]]);
		int closest_neighbor = ring_neighbors[n_v + i];
		for (int j = 1; j < ring_neighbors[i]; j++) {
			float curr_dist = sqrtDist(i_pos, m_pos[ring_neighbors[(j + 1) * n_v + i]]);
			if (curr_dist < closest_dist) {
				closest_dist = curr_dist;
				closest_neighbor = ring_neighbors[(j + 1) * n_v + i];
			}
		}
		__syncthreads();
		closest_neighbors[i] = closest_neighbor;
	}
}

template<int maxRingSize>
__global__ void compute_closestNeighbors_kernel2(const glm::vec2* m_pos, int n_v, const int* ring_neighbors, int* closest_neighbors) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_v) {
		glm::vec2 i_pos = m_pos[i];
		float closest_dist = sqrtDist(i_pos, m_pos[ring_neighbors[maxRingSize * i + 1]]);
		int closest_neighbor = ring_neighbors[maxRingSize * i + 1];
		for (int j = 1; j < ring_neighbors[maxRingSize*i]; j++) {
			float curr_dist = sqrtDist(i_pos, m_pos[ring_neighbors[(j + 1) * maxRingSize + i]]);
			if (curr_dist < closest_dist) {
				closest_dist = curr_dist;
				closest_neighbor = ring_neighbors[(j + 1) * maxRingSize + i];
			}
		}
		__syncthreads();
		closest_neighbors[i] = closest_neighbor;
	}
}


// n_he -> number of full edges
template<int maxRingSize, int maxFRNNSize>
__global__ void computeNeighbors_kernel(const glm::vec2* m_pos, int n_v, const int* ring_neighbors, int* neighbors, float rr) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_v) {
		glm::vec2 i_pos = m_pos[i];
		constexpr int stack_size = 64;
		int stack[stack_size];
		int stack_counter = 0;
		int neighbors_counter = 0;

		stack[stack_counter++] = i;

		while (stack_counter > 0) {

			int curr_neighbor_checking = stack[--stack_counter];

			// we start checking all the neighbors of the vertex we know is inside the FRNN
			for (int j = 0; j < ring_neighbors[curr_neighbor_checking]; j++) {

				int curr_vertex_checking = ring_neighbors[(j + 1) * n_v + curr_neighbor_checking];

				if (curr_vertex_checking == i)continue;

				bool is_already_a_neighbor = false;
				// Its faster not to check this for some reason
				//for (int k = 0; k < stack_counter && !is_already_a_neighbor; k++) {
				//	if (stack[k] == curr_vertex_checking)is_already_a_neighbor = true;
				//}
				for (int k = 0; k < neighbors_counter && !is_already_a_neighbor; k++) {
					if (neighbors[(k + 1) * n_v + i] == curr_vertex_checking) {
						is_already_a_neighbor = true;
					}
				}

				// if this happens, we know the vertex is a neighbor
				if (!is_already_a_neighbor && (sqrtDist(i_pos, m_pos[curr_vertex_checking]) <= rr)) {
					neighbors[(neighbors_counter + 1) * n_v + i] = curr_vertex_checking;
					neighbors_counter++;
					//if (neighbors_counter+1>=maxFRNNSize)break; // TODO: manage the neighbors overflow
					stack[stack_counter++] = curr_vertex_checking;
					//if (stack_counter>= stack_size)break; // TODO: manage the stack overflow
				}

			}
		}

		__syncthreads();

		neighbors[i] = neighbors_counter;
	}
}

// n_he -> number of full edges
template<int maxRingSize, int maxFRNNSize>
__global__ void computeNeighbors_kernel2(const glm::vec2* m_pos, int n_v, const int* ring_neighbors, int* neighbors, float rr) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_v) {
		glm::vec2 i_pos = m_pos[i];
		constexpr int stack_size = 64;
		int stack[stack_size];
		int stack_counter = 0;
		int neighbors_counter = 0;

		stack[stack_counter++] = i;

		while (stack_counter > 0) {

			int curr_neighbor_checking = stack[--stack_counter];

			// we start checking all the neighbors of the vertex we know is inside the FRNN
			for (int j = 0; j < ring_neighbors[curr_neighbor_checking* maxRingSize]; j++) {

				int curr_vertex_checking = ring_neighbors[(j + 1) + maxRingSize * curr_neighbor_checking];

				if (curr_vertex_checking == i)continue;

				bool is_already_a_neighbor = false;
				// Its faster not to check this for some reason
				//for (int k = 0; k < stack_counter && !is_already_a_neighbor; k++) {
				//	if (stack[k] == curr_vertex_checking)is_already_a_neighbor = true;
				//}
				for (int k = 0; k < neighbors_counter && !is_already_a_neighbor; k++) {
					if (neighbors[(k + 1) + maxFRNNSize * i] == curr_vertex_checking) {
						is_already_a_neighbor = true;
					}
				}

				// if this happens, we know the vertex is a neighbor
				if (!is_already_a_neighbor && (sqrtDist(i_pos, m_pos[curr_vertex_checking]) <= rr)) {
					neighbors[(neighbors_counter + 1) + maxFRNNSize * i] = curr_vertex_checking;
					neighbors_counter++;
					//if (neighbors_counter+1>=maxFRNNSize)break; // TODO: manage the neighbors overflow
					stack[stack_counter++] = curr_vertex_checking;
					//if (stack_counter>= stack_size)break; // TODO: manage the stack overflow
				}

			}
		}

		__syncthreads();

		neighbors[i* maxFRNNSize] = neighbors_counter;
	}
}

__global__ void flip_delaunay_kernel(const glm::vec2* m_pos, int* m_helper_t, int n_he, Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int* m_flag) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_he) {
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

		bool flip = ((t[0] >= 0) && (t[1] >= 0) && ((inCircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]]) > 0.0000001)) > 0 && (atomicExch(&m_helper_t[t[0]], i) == -1) && (atomicExch(&m_helper_t[t[1]], i) == -1));
		//bool flip = ((t[0]>=0) &&  (t[1]>=0) && (angle_incircle(m_pos[v[0]], m_pos[v[1]], m_pos[v[2]], m_pos[v[3]]) > 1.00001) && (atomicExch(&m_helper_t[t[0]], i) == -1) && (atomicExch(&m_helper_t[t[1]], i) == -1));

		if (flip) {
			f2to2(m_t, m_he, m_v, i * 2);
			*m_flag = 1;
		}
	}
}

__global__ void fix_triangles_kernel(const glm::vec2* m_pos, int* m_helper_t, int n_he, Triangle* m_t, HalfEdge* m_he, Vertex* m_v, int* m_flag) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_he) {
		bool flip = false;

		int he_index = i * 2;
		if (isCreased(m_t, m_he, m_pos, he_index)) {
			int he_upright = he_index;
			int he_inverted = he_index ^ 1;

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
				flip = ((atomicExch(&m_helper_t[t_index_upright], i) == -1) && (atomicExch(&m_helper_t[t_index_inverted], i) == -1));
			}

			// --------------------
			// C-Step
			// Case where the upright triangle is inside the inverted one
			else if (isInsideInverted(m_t, m_he, m_pos, op_upright, t_index_inverted)) {
				flip = ((atomicExch(&m_helper_t[t_index_upright], i) == -1) && (atomicExch(&m_helper_t[t_index_inverted], i) == -1));
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

			// --------------------
			// Others
			*m_flag = -1;
		}

		if (flip) {
			f2to2(m_t, m_he, m_v, i * 2);
			*m_flag = 1;
			//atomicCAS(m_flag, 0, 1);
		}
	}
}