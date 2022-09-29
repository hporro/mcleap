#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <random>

#include "Host_Triangulation.h"
#include "Helpers_Triangulation.h"
#include "Device_Triangulation.h"

typedef union {
    float f;
    struct {
        unsigned int mantisa : 23;
        unsigned int exponent : 8;
        unsigned int sign : 1;
    } parts;
} float_cast;

int main() {
	glm::vec2 a(0.0f, 3.3f);
    float_cast d1 = { 0.15625 };
    printf("sign = %x\n", d1.parts.sign);
    printf("exponent = %x\n", d1.parts.exponent);
    printf("mantisa = %x\n", d1.parts.mantisa);

}