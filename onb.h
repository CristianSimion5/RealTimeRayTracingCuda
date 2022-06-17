#pragma once

#include "vector3.h"

class onb {
public:
    __device__ onb() {}
    
    __device__ inline glm::vec3 operator[](int i) const { return axis[i]; }

    __device__ glm::vec3 u() const { return axis[0]; }
    __device__ glm::vec3 v() const { return axis[1]; }
    __device__ glm::vec3 w() const { return axis[2]; }

    __device__ glm::vec3 local(const glm::vec3& a) const {
        return a.x * axis[0] + a.y * axis[1] + a.z * axis[2];
    }

    __device__ void build_from_norm(const glm::vec3& n);

public:
    glm::vec3 axis[3];
};

__device__ void onb::build_from_norm(const glm::vec3& n) {
    axis[2] = glm::normalize(n);
    glm::vec3 a = (glm::abs(w().x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    axis[1] = glm::normalize(glm::cross(w(), a));
    axis[0] = glm::cross(w(), v());
}