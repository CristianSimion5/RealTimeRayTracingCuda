#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <cstdio>

#include "glm/gtc/constants.hpp"
#include "glm/trigonometric.hpp"
#include "glm/common.hpp"

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

using glm::radians;
using glm::clamp;

__device__ const float infinity = 1e9f;//std::numeric_limits<float>::infinity();
__device__ const float pi = 3.14159265358979323846264338327950288f;//glm::pi<float>();

__device__ inline float random_float(curandState* rand_state) {
    return curand_uniform(rand_state);
}

__device__ inline float random_float(float min, float max, curandState* rand_state) {
    return random_float(rand_state) * (max - min) + min;
}

__device__ inline int random_int(int max, curandState* rand_state) {
    return random_float(rand_state) * (max + 0.999999f);
}

inline float random_float() {
    static std::random_device rd;
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator(rd());
    return distribution(generator);
}

inline float random_float(float min, float max) {
    return random_float() * (max - min) + min;
}

inline int random_int() {
    static std::random_device rd;
    static std::uniform_int_distribution<int> int_distribution;
    static std::mt19937 generator(rd());
    return int_distribution(generator);
}

inline int random_int(int min, int max) {
    return random_int() % (max - min + 1) + min;
}

#include "ray.h"
#include "vector3.h"