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

#include "helper/stats.cuh"

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
	double movement = 0.0005;
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

	float* d_dt, * d_dt_helper, * h_dt = new float[4 + numP];
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

	std::vector<float> grid_update_data;
	std::vector<float> grid_closest_data;
	std::vector<float> delaunay_untangle_data;
	std::vector<float> delaunay_legalize_data;
	std::vector<float> delaunay_oneRing_data;
	std::vector<float> delaunay_closest_data;
	std::vector<float> delaunay_FRNN_data;
	std::vector<float> integrate_simulation_data;
	std::vector<float> points_moving_data;

	statistics  stats_grid_update;
	RunningStat meas_grid_update;
	statistics  stats_grid_closest_neighbor;
	RunningStat meas_grid_closest_neighbor;

	statistics  stats_delaunay_untangle;
	RunningStat meas_delaunay_untangle;
	statistics  stats_delaunay_legalize;
	RunningStat meas_delaunay_legalize;

	statistics  stats_delaunay_oneRing;
	RunningStat meas_delaunay_oneRing;
	statistics  stats_delaunay_closest_neighbor;
	RunningStat meas_delaunay_closest_neighbor;
	statistics  stats_delaunay_FRNN;
	RunningStat meas_delaunay_FRNN;

	statistics  stats_integrate_simulation;
	RunningStat meas_integrate_simulation;
	statistics  stats_points_moving;
	RunningStat meas_points_moving;

	for (int i = 0; i < 1000; i++) {
		//printf("\n");

		cudaMemcpy(d_pos, dt.m_pos, (4 + numP) * sizeof(glm::vec2), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();

		auto begin = std::chrono::high_resolution_clock::now();
		snfunctor->resetFunctor();
		gc.update(d_pos);
		auto end = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_grid_update.Push(diff);
		grid_update_data.push_back(diff);
		//printf("Update Grid: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		gc.apply_f_frnn<SaveNeighborsFunctor>(*snfunctor, d_pos, rad);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_grid_closest_neighbor.Push(diff);
		grid_closest_data.push_back(diff);
		//printf("Grid Frnn: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.untangle2();
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_delaunay_untangle.Push(diff);
		delaunay_untangle_data.push_back(diff);
		//printf("Delaunay untangle: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.delonize2();
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_delaunay_legalize.Push(diff);
		delaunay_legalize_data.push_back(diff);
		//printf("Delaunay legalize: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.oneRing<max_ring_neighbors>(d_ring_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_delaunay_oneRing.Push(diff);
		delaunay_oneRing_data.push_back(diff);
		//printf("Delaunay Ring: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.closestNeighbors <max_ring_neighbors>(d_ring_neighbors, d_closest_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_delaunay_closest_neighbor.Push(diff);
		delaunay_closest_data.push_back(diff);
		//printf("Delaunay Closest neighbor: %f\n", diff);

		begin = std::chrono::high_resolution_clock::now();
		dt.getFRNN<max_ring_neighbors, max_neighbors>(rad, d_ring_neighbors, d_neighbors);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_delaunay_FRNN.Push(diff);
		delaunay_FRNN_data.push_back(diff);
		//printf("Delaunay Frnn: %f\n", diff);

		dt.transferToHost();
		cudaMemcpy(h_neighbors, d_neighbors, (4 + numP) * max_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		begin = std::chrono::high_resolution_clock::now();
		dim3 dimBlock(128);
		dim3 dimGrid((numP + 4 + 128 - 1) / dimBlock.x);
		check_if_close_diffusion <max_neighbors> << <dimGrid, dimBlock >> > (numP + 4, dt.m_pos, d_neighbors, d_dt);
		dt.integrateMovePoints(d_dt, d_move);
		end = std::chrono::high_resolution_clock::now();
		diff = std::chrono::duration<float, std::milli>(end - begin).count();
		meas_integrate_simulation.Push(diff);
		integrate_simulation_data.push_back(diff);

		thrust::device_ptr<float> dt_array = thrust::device_pointer_cast(d_dt);
		thrust::device_ptr<float> dt_helper_array = thrust::device_pointer_cast(d_dt_helper);
		thrust::inclusive_scan(dt_array, dt_array + numP + 4, dt_helper_array);

		cudaDeviceSynchronize();
		float* res = new float[1];
		cudaMemcpy(res, &d_dt_helper[numP + 4 - 1], sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		meas_points_moving.Push(res[0]);
		points_moving_data.push_back(res[0]);
		//printf("There's %f particles moving\n", res[0]);
		delete[] res;

		//printf("Move points: %f\n", diff);
	}

	delete[] h_neighbors;
	delete[] h_dt;
	cudaFree(d_dt);
	cudaFree(d_dt_helper);
	cudaFree(d_pos);
	cudaFree(d_ring_neighbors);

#define PRINTDATA(arr_name) printf("%s = [",#arr_name);for (int i = 0; i < arr_name.size(); i++) {printf("%f, ", arr_name[i]);}printf("]\n");
	PRINTDATA(grid_update_data);
	PRINTDATA(grid_closest_data);
	PRINTDATA(delaunay_untangle_data);
	PRINTDATA(delaunay_legalize_data);
	PRINTDATA(delaunay_oneRing_data);
	PRINTDATA(delaunay_closest_data);
	PRINTDATA(delaunay_FRNN_data);
	PRINTDATA(integrate_simulation_data);
	PRINTDATA(points_moving_data);
	//std::vector<float> grid_update_data;           
	//std::vector<float> grid_closest_data;          
	//std::vector<float> delaunay_untangle_data;    
	//std::vector<float> delaunay_legalize_data;    
	//std::vector<float> delaunay_oneRing_data;     
	//std::vector<float> delaunay_closest_data;     
	//std::vector<float> delaunay_FRNN_data;        
	//std::vector<float> integrate_simulation_data;
	//std::vector<float> points_moving_data;       

	return 0;
}