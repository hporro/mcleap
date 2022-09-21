#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

struct cmp_points {
	bool operator()(glm::vec2& a, glm::vec2& b) {
		if (a.x < b.x)return false;
		if (a.y < b.y)return false;
		return true;
	}
};

int main(int argc, char* argv[]) {
	int numP = 100000; // ~2:20[min] for 10e6, ~1.6[s] for 10e5 (in Release mode in the notebook)
	double bounds = 10000.0;
	double movement = 0.1;
	if (argc > 1) {
		numP = atoi(argv[1]);
	}
	if (argc > 2) {
		bounds = atof(argv[2]);
	}
	if (argc > 3) {
		movement = atof(argv[3]);
	}

	std::vector<glm::vec2> h_pos, h_move;
	glm::vec2* d_move;
	cudaMalloc((void**)&d_move, numP * sizeof(glm::vec2));
	std::random_device dev;
	std::mt19937 rng{ dev() };;

	std::uniform_real_distribution<float> pos_r(-bounds, bounds);
	std::uniform_real_distribution<float> move_r(-movement, movement);

	for (int i = 0; i < numP; i++) {
		h_pos.push_back(glm::vec2(pos_r(rng), pos_r(rng)));
	}
	std::sort(h_pos.begin(), h_pos.end(), cmp_points{});
	for (int i = 0; i < numP; i++) {
		h_move.push_back(glm::vec2(move_r(rng), move_r(rng)));
	}
	cudaMemcpy(d_move, h_move.data(), numP * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	HostTriangulation* ht = new HostTriangulation();
	for (auto p : h_pos) {
		ht->addDelaunayPoint(p);
	}
	
	DeviceTriangulation dt(ht);
	dt.untangle();
	dt.delonize();
	for (int i = 0; i < 1000; i++) {
		dt.movePoints(d_move);
		dt.untangle();
		dt.delonize();
		//int non_delaunay_edges_count = 0;
		//int creased_edges_count = 0;
		//int inverted_edges_count = 0;
		////checking delaunay condition
		//dt.transferToHost();
		//for (int i = 0; i < ht->m_he.size() / 2; i++) {
		//	int v[4];
		//	HalfEdge he[4];
		//	int t[2];
		//	t[0] = ht->m_he[i * 2].t;
		//	t[1] = ht->m_he[i * 2 ^ 1].t;
		//	if (t[0] * t[1] < 0)continue; // if one of them is negative (convex hull of the mesh) doesnt count
		//	he[0] = ht->m_he[i * 2];
		//	he[1] = ht->m_he[he[0].next];
		//	he[2] = ht->m_he[he[1].next];
		//	he[3] = ht->m_he[i * 2 ^ 1];
		//	v[0] = he[0].v;
		//	v[1] = ht->m_he[ht->m_he[ht->m_he[i * 2 ^ 1].next].next].v;
		//	v[2] = he[1].v;
		//	v[3] = he[2].v;
		//
		//	//bool flag = false;
		//
		//	//for (int i = 0; i < 4 && !flag; i++) {
		//	//	//check convexity of the bicell
		//	//	if (!isToTheLeft(ht->m_pos[v[i]], ht->m_pos[v[(i + 1) % 4]], ht->m_pos[v[(i + 2) % 4]]))flag = true;
		//	//}
		//
		//	if (inCircle(ht->m_pos[v[0]], ht->m_pos[v[1]], ht->m_pos[v[2]], ht->m_pos[v[3]]))non_delaunay_edges_count++;
		//	if (isCreased(ht->m_t.data(), ht->m_he.data(), ht->m_pos.data(), i * 2))creased_edges_count++;
		//	if (isInvertedEdge(ht->m_t.data(), ht->m_he.data(), ht->m_pos.data(), i * 2))inverted_edges_count++;
		//
		//}
		//
		//printf("Number of vertices: %d\n", ht->m_pos.size());
		//printf("Non-delaunay edges: %d\n", non_delaunay_edges_count);
		//printf("Creased edges: %d\n", creased_edges_count);
		//printf("Inverted edges: %d\n", inverted_edges_count);
	}

	return 0;
}