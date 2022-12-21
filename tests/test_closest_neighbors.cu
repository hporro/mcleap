#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

#include <tinytest.h>

template<int num_vertices, int max_ring_neighbors>
void test_closest_neighbors(double movement, double bounds) {
	int numP = num_vertices;

	std::vector<glm::vec2> h_pos, h_move;
	glm::vec2* d_move;
	cudaMalloc((void**)&d_move, numP * sizeof(glm::vec2));
	//std::random_device dev;
	std::mt19937 rng{ 3 };

	std::uniform_real_distribution<float> pos_r(-bounds, bounds);
	std::uniform_real_distribution<float> move_r(-movement, movement);

	for (int i = 0; i < numP; i++) {
		h_pos.push_back(glm::vec2(pos_r(rng), pos_r(rng)));
	}
	for (int i = 0; i < numP; i++) {
		h_move.push_back(glm::vec2(move_r(rng), move_r(rng)));
	}
	cudaMemcpy(d_move, h_move.data(), numP * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	HostTriangulation* ht = new HostTriangulation();
	ht->addDelaunayPoints(h_pos);

	int* d_ring_neighbors, * h_ring_neighbors;
	cudaMalloc((void**)&d_ring_neighbors, ht->m_pos.size() * max_ring_neighbors * sizeof(int));

	int* d_closest_neighbors, * h_closest_neighbors = new int[ht->m_pos.size()];
	cudaMalloc((void**)&d_closest_neighbors, ht->m_pos.size() * sizeof(int));

	DeviceTriangulation dt(ht);
	//dt.untangle();
	//dt.delonize();
	dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
	dt.closestNeighbors<max_ring_neighbors>(d_ring_neighbors, d_closest_neighbors);

	cudaMemcpy(h_closest_neighbors, d_closest_neighbors, ht->m_pos.size() * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int* real_closest_neighbors = new int[ht->m_pos.size()];
	//memset(real_neighbors, 0, ht->m_pos.size() * sizeof(int));
	for (int i = 0; i < ht->m_pos.size(); i++)real_closest_neighbors[i] = -1;

	for (int i = 0; i < ht->m_pos.size(); i++) {
		glm::vec2 i_pos = ht->m_pos[i];
		float closest_dist = sqrtDist(i_pos, ht->m_pos[0]); if (i == 0)closest_dist = sqrtDist(i_pos, ht->m_pos[1]);
		int closest_neighbor = 0; if (i == 0)closest_neighbor = 1;

		for (int j = 0; j < ht->m_pos.size(); j++) {
			if (i == j)continue;
			float curr_dist = sqrtDist(i_pos, ht->m_pos[j]);
			if (curr_dist < closest_dist) {
				closest_dist = curr_dist;
				closest_neighbor = j;
			}
		}
		real_closest_neighbors[i] = closest_neighbor;
	}

	//std::cout << "REAL NEIGHBORS" << std::endl;
	//for (int i = 0; i < ht->m_pos.size(); i++) {
	//	if (i == 185)std::cout << "i: " << i << " num_neighbors: " << real_neighbors[i] << " neighbors: ";
	//	for (int j = 1; j <= real_neighbors[i]; j++) {
	//		if (i == 185)std::cout << real_neighbors[ht->m_pos.size() * j + i] << " ";
	//	}
	//	if (i == 185)std::cout << std::endl;
	//}
	//
	//std::cout << "GPU NEIGHBORS" << std::endl;
	//for (int i = 0; i < ht->m_pos.size(); i++) {
	//	if (i == 185)std::cout << "i: " << i << " num_neighbors: " << h_neighbors[i] << " neighbors: ";
	//	for (int j = 1; j <= h_neighbors[i]; j++) {
	//		if (i == 185)std::cout << h_neighbors[ht->m_pos.size() * j + i] << " ";
	//	}
	//	if (i == 185)std::cout << std::endl;
	//}

	for (int i = 0; i < ht->m_pos.size(); i++) {
		//if(i==185)std::cout << "GPU  neighbors vertex num " << i << ": " << h_neighbors[i] << std::endl;
		//if(i==185)std::cout << "Real neighbors vertex num " << i << ": " << real_neighbors[i] << std::endl;

		// For now, I'm just checking number of neighbors, not if the neighbors are actually the same ones 
		//printf("i: %d real: %d computed: %d\n", i, real_closest_neighbors[i], h_closest_neighbors[i]);
		ASSERT_EQUALS(real_closest_neighbors[i], h_closest_neighbors[i]);
	}

	delete ht;
	delete[] h_closest_neighbors;
	delete[] real_closest_neighbors;
	cudaFree(d_move);
	cudaFree(d_ring_neighbors);
	cudaFree(d_closest_neighbors);
}

int main(int argc, char* argv[]) {
	RUN((test_closest_neighbors<10, 100>), 0.1, 1000.0);
	RUN((test_closest_neighbors<100, 100>), 0.1, 1000.0);
	RUN((test_closest_neighbors<1000, 100>), 0.1, 1000.0);
	return TEST_REPORT();
}