#pragma once

#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

#include "vector3.h"

class ray {
public:
    __device__ ray() = default;
    __device__ ray(const point3& origin, const glm::vec3& direction, float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}

    __device__ point3 origin() const       { return orig; }
    __device__ glm::vec3 direction() const { return dir; }
    __device__ float time() const          { return tm; }

    __device__ point3 at(float t) const {
        return orig + t * dir;
    }

public:
    point3 orig;
    glm::vec3 dir;
    float tm;
};

