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
	std::random_device dev();
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
	cudaMalloc((void**)&d_ring_neighbors, numP * max_ring_neighbors * sizeof(int));

	int* d_neighbors, * h_neighbors = new int[numP * max_neighbors];
	cudaMalloc((void**)&d_neighbors, numP * max_neighbors * sizeof(int));

	DeviceTriangulation dt(ht);
	//dt.untangle();
	//dt.delonize();
	dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
	dt.getFRNN<max_ring_neighbors, max_neighbors>(radius, d_ring_neighbors, d_neighbors);

	cudaMemcpy(h_neighbors, d_neighbors, numP * max_neighbors * sizeof(int), cudaMemcpyDeviceToHost);

	float rr = radius * radius;
	int* real_neighbors = new int[numP];
	memset(real_neighbors, 0, numP * sizeof(int));

	std::cout << std::endl;
	std::cout << h_pos[0].x << " " << h_pos[0].y << std::endl;
	std::cout << h_pos[1].x << " " << h_pos[1].y;
	std::cout << std::endl;

	for (int i = 0; i < numP; i++) {
		for (int j = i+1; j < numP; j++) {
			if (sqrtDist(h_pos[i], h_pos[j]) <= rr) {
				real_neighbors[i]++;
				real_neighbors[j]++;
			}
		}
	}

	for (int i = 0; i < numP; i++) {
		std::cout << "Num  neighbors vertex num " << i << ": " << h_neighbors[i+4] << " " << h_neighbors[max_neighbors*1+i+4] << std::endl;
		std::cout << "Real neighbors vertex num " << i << ": " << real_neighbors[i] << std::endl;
		ASSERT_EQUALS(real_neighbors[i], h_neighbors[i+4]);
	}

	delete[] h_neighbors;
	delete[] real_neighbors;
	cudaFree(d_move);
	cudaFree(d_ring_neighbors);
	cudaFree(d_neighbors);
}

int main(int argc, char* argv[]){
	RUN((test_frnn<100,10,100>),0.1,100.0,10);
	return TEST_REPORT();
}