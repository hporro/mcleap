#pragma once

#include "Common_Triangulation.h"

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
	if (i < n) {
		pos[i]+=d[i];
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

__device__ __host__ float sqrtDist(const glm::vec2& a, const glm::vec2& b) {
	return pow2(a.x-b.x) + pow2(a.y-b.y);
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
