#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

#include <tinytest.h>

template<int num_vertices, int max_ring_neighbors, int max_neighbors>
void test_frnn(double movement, double bounds, float radius) {
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

	int* d_neighbors, * h_neighbors = new int[ht->m_pos.size() * max_neighbors];
	cudaMalloc((void**)&d_neighbors, (numP+4) * max_neighbors * sizeof(int));

	DeviceTriangulation dt(ht);
	//dt.untangle();
	//dt.delonize();
	dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
	dt.getFRNN<max_ring_neighbors, max_neighbors>(radius, d_ring_neighbors, d_neighbors);
	//dt.oneRing2<max_ring_neighbors>(d_ring_neighbors);
	//dt.getFRNN2<max_ring_neighbors, max_neighbors>(radius, d_ring_neighbors, d_neighbors);

	cudaMemcpy(h_neighbors, d_neighbors, (numP+4) * max_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	float rr = radius * radius;
	int* real_neighbors = new int[ht->m_pos.size() * max_neighbors];
	//memset(real_neighbors, 0, ht->m_pos.size() * sizeof(int));
	for (int i = 0; i < ht->m_pos.size() * max_neighbors; i++)real_neighbors[i] = 0;

	for (int i = 0; i < ht->m_pos.size(); i++) {
		for (int j = i+1; j < ht->m_pos.size(); j++) {
			if (sqrtDist(ht->m_pos[i], ht->m_pos[j]) <= rr) {
				//printf("Neighbors: i: %d j: %d\n", i, j);
				real_neighbors[i]++;
				real_neighbors[j]++;
				real_neighbors[ht->m_pos.size() * real_neighbors[i] + i] = j;
				real_neighbors[ht->m_pos.size() * real_neighbors[j] + j] = i;
			}
		}
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
		ASSERT_EQUALS(real_neighbors[i], h_neighbors[i]);
		//ASSERT_EQUALS(real_neighbors[i], h_neighbors[i * max_neighbors]); // -> in order to check oneRing2 and getFRNN2
	}

	delete ht;
	delete[] h_neighbors;
	delete[] real_neighbors;
	cudaFree(d_move);
	cudaFree(d_ring_neighbors);
	cudaFree(d_neighbors);
}

int main(int argc, char* argv[]){
	RUN((test_frnn<1000,20,100>),0.1,1000.0,5.0f);
	RUN((test_frnn<1000,20,100>),0.1,1000.0,10.0f);
	RUN((test_frnn<1000,20,100>),0.1,1000.0,20.0f);
	RUN((test_frnn<1000,20,100>),0.1,1000.0,30.0f);
	RUN((test_frnn<1000,20,100>),0.1,1000.0,40.0f);
	RUN((test_frnn<1000,20,100>),0.1,1000.0,50.0f);
	return TEST_REPORT();
}