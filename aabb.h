#pragma once

#include "utility.h"

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(const point3& a, const point3& b) { minimum = a, maximum = b; }

    __device__ point3 min() const { return minimum; }
    __device__ point3 max() const { return maximum; }

    __device__ bool hit(const ray& r, float t_min, float t_max, curandState* rng = nullptr) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.f / r.direction()[a];
            auto t0 = (min()[a] - r.origin()[a]) * invD;
            auto t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.f) {
                // std::swap(t0, t1);
                auto temp = t0;
                t0 = t1;
                t1 = temp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

public:
    point3 minimum;
    point3 maximum;
};

__device__ aabb surrounding_box(const aabb& box0, const aabb& box1) {
    point3 small(glm::min(box0.min().x, box1.min().x),
                 glm::min(box0.min().y, box1.min().y),
                 glm::min(box0.min().z, box1.min().z));

    point3 large(glm::max(box0.max().x, box1.max().x),
                 glm::max(box0.max().y, box1.max().y),
                 glm::max(box0.max().z, box1.max().z));

    return aabb(small, large);
}

