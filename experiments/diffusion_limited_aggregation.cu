#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"
#include "gridCount2d.h"

struct SaveNeighborsFunctor {
	SaveNeighborsFunctor(float rad, int numP, int max_neighbors) : m_numP(numP), h_m_max_neighbors(max_neighbors) {
		cudaMalloc((void**)&m_rad, sizeof(float));
		cudaMalloc((void**)&m_max_neighbors, sizeof(int));
		cudaMalloc((void**)&m_num_neighbors, numP * sizeof(int));
		cudaMalloc((void**)&m_neighbors, h_m_max_neighbors * numP * sizeof(int));

		cudaMemcpy(m_rad, &rad, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(m_max_neighbors, &max_neighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	~SaveNeighborsFunctor() {
		cudaFree(m_rad);
		cudaFree(m_max_neighbors);
		cudaFree(m_num_neighbors);
		cudaFree(m_neighbors);
	}
	void resetFunctor() {
		cudaMemset(m_num_neighbors, 0, m_numP * sizeof(int));
		cudaMemset(m_neighbors, 0, m_numP * h_m_max_neighbors * sizeof(int));
		cudaDeviceSynchronize();
	}
	__device__ void operator()(const int& i, const int& j, const glm::vec2 dist_vec, const double dist) {
		if (i != j)if (dist <= *m_rad) {
			int ind = i * (*m_max_neighbors) + m_num_neighbors[i];
			if (ind < m_numP * (*m_max_neighbors))m_neighbors[ind] = j;
			m_num_neighbors[i]++;
		}
	}
	int m_numP, h_m_max_neighbors;
	float* m_rad;
	int* m_max_neighbors;
	int* m_num_neighbors;
	int* m_neighbors;
};

template<int maxFRNN>
__global__ void check_if_close_diffusion(int n, glm::vec2* m_pos, int* neighbors, float* dt) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && i >= 4) { // to not move 4 bounding points
		for (int j = 0; j < neighbors[i]; j++) {
			int neighbor = neighbors[i + (j + 1) * n];
			if (dt[neighbor] == 0.0f)dt[i] = 0.0f;
		}
	}
}

int main(int argc, char* argv[]) {
	int numP = 1000000; // ~8[s] for 10e6, ~2[s] for 10e5 (in Debug mode in the office while doing profiling and diagnostics tool)
	double bounds = 10000.0;
	double movement = 0.005;
	constexpr float rad = 10.f;
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
	glm::vec2* d_move, * d_pos;
	cudaMalloc((void**)&d_move, (4 + numP) * sizeof(glm::vec2));
	cudaMalloc((void**)&d_pos, (4 + numP) * sizeof(glm::vec2));

	std::random_device dev;
	std::mt19937 rng{ dev() };

	std::uniform_real_distribution<float> pos_r(-bounds, bounds);
	std::uniform_real_distribution<float> move_r(-movement, movement);

	for (int i = 0; i < numP; i++) {
		h_pos.push_back(glm::vec2(pos_r(rng), pos_r(rng)));
	}
	for (int i = 0; i < numP + 4; i++) {
		h_move.push_back(glm::vec2(move_r(rng), move_r(rng)));
	}
	cudaMemcpy(d_move, h_move.data(), (4 + numP) * sizeof(glm::vec2), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	HostTriangulation* ht = new HostTriangulation();
	ht->addDelaunayPoints(h_pos);

	printf("Triangulated (in the host)!!\n");

	int* d_ring_neighbors, * h_ring_neighbors;
	constexpr int max_ring_neighbors = 10;
	cudaMalloc((void**)&d_ring_neighbors, numP * max_ring_neighbors * sizeof(int));
	int* d_neighbors, * h_neighbors;
	constexpr int max_neighbors = 200;
	cudaMalloc((void**)&d_neighbors, (4 + numP) * max_neighbors * sizeof(int));
	h_neighbors = new int[(4 + numP) * max_neighbors];

	int* d_closest_neighbors, * h_closest_neighbors;
	cudaMalloc((void**)&d_closest_neighbors, (4 + numP) * sizeof(int));
	h_closest_neighbors = new int[4 + numP];

	DeviceTriangulation dt(ht);

	GridCount2d gc(numP + 4, glm::vec2(-bounds, -bounds), glm::vec2(rad, rad), glm::ivec2(ceil(2.0 * bounds / rad), ceil(2.0 * bounds / rad)));
	SaveNeighborsFunctor* snfunctor = new SaveNeighborsFunctor(rad, 4 * numP, max_neighbors);
	snfunctor->resetFunctor();
	cudaMemcpy(d_pos, dt.m_pos, (4 + numP) * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	
	float* d_dt, *d_dt_helper, * h_dt = new float[4+numP];
	cudaMalloc((void**)&d_dt, (4 + numP) * sizeof(float));
	cudaMalloc((void**)&d_dt_helper, (4 + numP) * sizeof(float));
	for (int i = 0; i < 4 + numP; i++) {
		h_dt[i] = 1.0f;
		if (i % 2 == 0)h_dt[i] = 0.0f;
	}
	h_dt[0] = 0.0f;
	h_dt[1] = 0.0f;
	h_dt[2] = 0.0f;
	h_dt[3] = 0.0f;
	cudaMemcpy(d_dt, h_dt, (4 + numP) * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	for (int i = 0; i < 100; i++) {
		printf("\n");


		cudaMemcpy(d_pos, dt.m_pos, (4 + numP) * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();

		auto begin = std::chrono::high_resolution_clock::now();
		snfunctor->resetFunctor();
		gc.update(d_pos);
		auto end = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("Update Grid: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		gc.apply_f_frnn<SaveNeighborsFunctor>(*snfunctor, d_pos, rad);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("Grid Frnn: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.untangle2();
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		printf("Delaunay untangle: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.delonize2();
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("Delaunay legalize: %f\n", diff);

		printf("Close indexing:\n");

		begin = std::chrono::high_resolution_clock::now();
		dt.oneRing2<max_ring_neighbors>(d_ring_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("\tDelaunay Ring: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.closestNeighbors2 <max_ring_neighbors>(d_ring_neighbors, d_closest_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		 
		printf("\tDelaunay Closest neighbor: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.getFRNN2<max_ring_neighbors, max_neighbors>(rad, d_ring_neighbors, d_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("\tDelaunay Frnn: %f\n", diff);
		printf("Far indexing:\n");

		begin = std::chrono::high_resolution_clock::now();
		dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("\tDelaunay Ring: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.closestNeighbors <max_ring_neighbors>(d_ring_neighbors, d_closest_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("\tDelaunay Closest neighbor: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.getFRNN<max_ring_neighbors, max_neighbors>(rad, d_ring_neighbors, d_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();

		printf("\tDelaunay Frnn: %f\n", diff);

		dt.transferToHost();
		cudaMemcpy(h_neighbors, d_neighbors, (4 + numP) * max_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		float avg_num = 0;
		for (int i = 0; i < 4 + numP; i++) {
			avg_num += h_neighbors[i];
		}
		avg_num /= 4 + numP;

		int non_delaunay_edges_count_matrix = 0;
		int non_delaunay_edges_count_angles = 0;
		int creased_edges_count = 0;
		int inverted_edges_count = 0;
		//checking delaunay condition
		for (int i = 0; i < ht->m_he.size() / 2; i++) {
			int v[4];
			int t[2];
			t[0] = ht->m_he[i * 2].t;
			t[1] = ht->m_he[i * 2 ^ 1].t;

			if (t[0] == -1 || t[1] == -1)continue; // if one of them is negative (convex hull of the mesh) doesnt count

			v[0] = ht->m_he[i * 2].v;
			v[1] = ht->m_he[i * 2 ^ 1].op;
			v[2] = ht->m_he[i * 2 ^ 1].v;
			v[3] = ht->m_he[i * 2].op;

			if (inCircle(ht->m_pos[v[0]], ht->m_pos[v[1]], ht->m_pos[v[2]], ht->m_pos[v[3]]) > 0)non_delaunay_edges_count_matrix++;
			if (angle_incircle(ht->m_pos[v[0]], ht->m_pos[v[1]], ht->m_pos[v[2]], ht->m_pos[v[3]]) > 1.00001)non_delaunay_edges_count_angles++;
			if (isCreased(ht->m_t.data(), ht->m_he.data(), ht->m_pos.data(), i * 2))creased_edges_count++;
			if (isInvertedEdge(ht->m_t.data(), ht->m_he.data(), ht->m_pos.data(), i * 2))inverted_edges_count++;

		}

		printf("Average number of neighbors: %f\n", avg_num);
		printf("Number of vertices: %d\n", ht->m_pos.size());
		printf("Non-delaunay edges matrix: %d\n", non_delaunay_edges_count_matrix);
		printf("Non-delaunay edges angles: %d\n", non_delaunay_edges_count_angles);
		printf("Creased edges: %d\n", creased_edges_count);
		printf("Inverted edges: %d\n", inverted_edges_count);

		for (int i = 0; i < numP + 4; i++) {
			h_move.push_back(glm::vec2(move_r(rng), move_r(rng)));
		}
		cudaMemcpy(d_move, h_move.data(), (numP + 4) * sizeof(glm::vec2), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		begin = std::chrono::high_resolution_clock::now();
		dim3 dimBlock(128);
		dim3 dimGrid((numP + 4 + 128 - 1) / dimBlock.x);
		check_if_close_diffusion <max_neighbors> << <dimGrid, dimBlock >> > (numP + 4, dt.m_pos, d_neighbors, d_dt);
		dt.integrateMovePoints(d_dt, d_move);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();


		thrust::device_ptr<float> dt_array = thrust::device_pointer_cast(d_dt);
		thrust::device_ptr<float> dt_helper_array = thrust::device_pointer_cast(d_dt_helper);
		thrust::inclusive_scan(dt_array, dt_array + numP + 4, dt_helper_array);

		cudaDeviceSynchronize();
		float* res = new float[1];
		cudaMemcpy(res, &d_dt_helper[numP+4-1], sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		printf("There's %f particles moving\n", res[0]);
		delete[] res;

		printf("Move points: %f\n", diff);

	}

	delete[] h_neighbors;
	delete[] h_dt;
	cudaFree(d_dt);
	cudaFree(d_dt_helper);
	cudaFree(d_pos);
	cudaFree(d_ring_neighbors);

	return 0;
}