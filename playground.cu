#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include <libmorton/morton.h>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

struct cmp_points {
	bool operator()(glm::vec2& a, glm::vec2& b) {
		uint_fast32_t am = libmorton::m2D_e_LUT< uint_fast32_t, uint_fast32_t>(a.x, a.y);
		uint_fast32_t bm = libmorton::m2D_e_LUT< uint_fast32_t, uint_fast32_t>(b.x, b.y);
		return am < bm;
		return true;
	}
};

int main(int argc, char* argv[]) {
	int numP = 1000000; // ~8[s] for 10e6, ~2[s] for 10e5 (in Debug mode in the office while doing profiling and diagnostics tool)
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
	std::random_device dev();
	std::mt19937 rng{ 3 };

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
	printf("Triangulated (in the host)!!\n");

	int* d_ring_neighbors, * h_ring_neighbors;
	constexpr int max_ring_neighbors = 10;
	cudaMalloc((void**)&d_ring_neighbors, numP * max_ring_neighbors * sizeof(int));
	int* d_neighbors, * h_neighbors;
	constexpr int max_neighbors = 200;
	cudaMalloc((void**)&d_neighbors, numP * max_neighbors * sizeof(int));

	DeviceTriangulation dt(ht);
	dt.untangle();
	dt.delonize();
	dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
	dt.getFRNN<max_ring_neighbors,max_neighbors>(70.f,d_ring_neighbors,d_neighbors);
	for (int i = 0; i < 2; i++) {
		dt.movePoints(d_move);
		dt.untangle();
		dt.delonize();
		dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
		dt.getFRNN<max_ring_neighbors, max_neighbors>(70.f, d_ring_neighbors, d_neighbors);
		int non_delaunay_edges_count = 0;
		int creased_edges_count = 0;
		int inverted_edges_count = 0;
		//checking delaunay condition
		dt.transferToHost();
		for (int i = 0; i < ht->m_he.size() / 2; i++) {
			int v[4];
			HalfEdge he[4];
			int t[2];
			t[0] = ht->m_he[i * 2].t;
			t[1] = ht->m_he[i * 2 ^ 1].t;
			if (t[0] * t[1] < 0)continue; // if one of them is negative (convex hull of the mesh) doesnt count
			he[0] = ht->m_he[i * 2];
			he[1] = ht->m_he[he[0].next];
			he[2] = ht->m_he[he[1].next];
			he[3] = ht->m_he[i * 2 ^ 1];
			v[0] = he[0].v;
			v[1] = ht->m_he[ht->m_he[ht->m_he[i * 2 ^ 1].next].next].v;
			v[2] = he[1].v;
			v[3] = he[2].v;
		
			if (inCircle(ht->m_pos[v[0]], ht->m_pos[v[1]], ht->m_pos[v[2]], ht->m_pos[v[3]]))non_delaunay_edges_count++;
			if (isCreased(ht->m_t.data(), ht->m_he.data(), ht->m_pos.data(), i * 2))creased_edges_count++;
			if (isInvertedEdge(ht->m_t.data(), ht->m_he.data(), ht->m_pos.data(), i * 2))inverted_edges_count++;
		
		}
		
		printf("Number of vertices: %d\n", ht->m_pos.size());
		printf("Non-delaunay edges: %d\n", non_delaunay_edges_count);
		printf("Creased edges: %d\n", creased_edges_count);
		printf("Inverted edges: %d\n", inverted_edges_count);
	}

	return 0;
}