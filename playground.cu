#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <delaunator.hpp>
#include <glm/glm.hpp>
#include <random>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

int main() {
	constexpr int numP = 1000;
	constexpr int blocksize = 64;
	std::vector<double> h_pos;
	std::random_device dev;
	std::mt19937 rng{ dev() };;
	//rng.seed(10);
	std::uniform_real_distribution<> distx(-1.0, 1.0);

	for (int i = 0; i < 2 * numP; i++) {
		h_pos.push_back(distx(rng));
	}
	h_pos[0] = -1000.0;
	h_pos[1] = -1000.0;
	h_pos[2] = 1000.0;
	h_pos[3] = -1000.0;
	h_pos[4] = 1000.0;
	h_pos[5] = 1000.0;
	h_pos[6] = -1000.0;
	h_pos[7] = 1000.0;

	delaunator::Delaunator d(h_pos);

	std::vector<double> coords = d.coords;

	std::vector<int> triangles;
	for (int i = 0; i < d.triangles.size(); i++) {
		triangles.push_back(d.triangles[i]);
	}
	std::vector<int> halfedges;
	for (int i = 0; i < d.halfedges.size(); i++) {
		halfedges.push_back(d.halfedges[i]);
	}

	coords[0] = 0.0;
	coords[1] = 0.0;

}