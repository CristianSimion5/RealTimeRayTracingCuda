#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ostream>

#include "glm/vec3.hpp"
#include "glm/geometric.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtc/random.hpp"
#include "glm/gtc/epsilon.hpp"

using color = glm::vec3;
using point3 = glm::vec3;

__device__ bool near_zero(const glm::vec3& s) {
    const float eps = 1e-8f;
    return glm::all(glm::epsilonEqual(s, glm::vec3(0.0f), eps));
}

__device__ inline static glm::vec3 random(curandState* rand_state) {
    return glm::vec3(random_float(rand_state), random_float(rand_state), random_float(rand_state));
}

__device__ inline static glm::vec3 random(float min, float max, curandState* rand_state) {
    return glm::vec3(random_float(min, max, rand_state), 
                     random_float(min, max, rand_state), 
                     random_float(min, max, rand_state));
}

__device__ glm::vec3 random_in_unit_disk(curandState* rand_state) {
    while (true) {
        auto p = glm::vec3(random_float(-1, 1, rand_state), random_float(-1, 1, rand_state), 0);
        if (glm::length2(p) >= 1.0f) continue;
        return p;
    }
}

__device__ inline glm::vec3 random_in_unit_sphere(curandState* rand_state) {
    while (true) {
        auto p = random(-1.0f, 1.0f, rand_state);
        if (glm::length2(p) >= 1.0f) continue;
        return p;
    }
}

__device__ glm::vec3 random_unit_vector(curandState* rand_state) {
    return glm::normalize(random_in_unit_sphere(rand_state));
}

__device__ glm::vec3 random_in_hemisphere(const glm::vec3& normal, curandState* rand_state) {
    glm::vec3 in_unit_sphere = random_in_unit_sphere(rand_state);

    // Verificam daca directia generata este in aceeasi emisfera cu normala
    if (glm::dot(in_unit_sphere, normal) > 0.f)
        return in_unit_sphere;
    return -in_unit_sphere;
}

std::ostream& operator<<(std::ostream& out, const glm::vec3& vec) {
    out << '(' << vec.x << ' ' << vec.y << ' ' << vec.z  << ')';
    return out;
}