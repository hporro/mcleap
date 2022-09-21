#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

int main() {
	constexpr int numP = 10000;

	std::vector<glm::vec2> h_pos, h_move;
	glm::vec2* d_move;
	cudaMalloc((void**)&d_move, numP * sizeof(glm::vec2));
	std::random_device dev;
	std::mt19937 rng{ dev() };;

	std::uniform_real_distribution<float> pos_r(-100.0, 100.0);
	std::uniform_real_distribution<float> move_r(-0.1, 0.1);

	for (int i = 0; i < 2 * numP; i++) {
		h_pos.push_back(glm::vec2(pos_r(rng), pos_r(rng)));
	}
	for (int i = 0; i < 2 * numP; i++) {
		h_move.push_back(glm::vec2(move_r(rng), move_r(rng)));
	}
	cudaMemcpy(d_move, h_move.data(), numP * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	HostTriangulation* ht = new HostTriangulation();
	for (auto p : h_pos) {
		ht->addPoint(p); // non-delaunay on porpoise
	}
	
	DeviceTriangulation dt(ht);
	dt.untangle();
	dt.delonize();
	dt.movePoints(d_move);
	dt.untangle();
	dt.delonize();

	//checking delaunay condition
	dt.transferToHost();
	for (int i = 0; i < ht->m_he.size()/2; i++) {
		int v[4];
		HalfEdge he[4];
		he[0] = ht->m_he[i*2];
		he[1] = ht->m_he[he[0].next];
		he[2] = ht->m_he[he[1].next];
		he[3] = ht->m_he[i*2 ^ 1];
		v[0] = he[0].v;
		v[1] = ht->m_he[ht->m_he[ht->m_he[i*2 ^ 1].next].next].v;
		v[2] = he[1].v;
		v[3] = he[2].v;
		
		//bool flag = false;

		//for (int i = 0; i < 4 && !flag; i++) {
		//	//check convexity of the bicell
		//	if (!isToTheLeft(ht->m_pos[v[i]], ht->m_pos[v[(i + 1) % 4]], ht->m_pos[v[(i + 2) % 4]]))flag = true;
		//}

		if (inCircle(ht->m_pos[v[0]], ht->m_pos[v[1]], ht->m_pos[v[2]], ht->m_pos[v[3]]))printf("We have a problem :(\n");
	}
	return 0;
}