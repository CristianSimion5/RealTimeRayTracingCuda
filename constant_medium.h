#pragma once

#include "utility.h"

#include "hittable.h"
#include "material.h"
#include "texture.h"

class constant_medium : public hittable {
public:
    __device__ constant_medium(hittable* b, float d, _texture* a)
        : boundary(b),
          neg_inv_density(-1.0f / d),
          phase_function(new isotropic(a))
        {}

    __device__ constant_medium(hittable* b, float d, color c)
        : boundary(b),
          neg_inv_density(-1.0f / d),
          phase_function(new isotropic(c))
        {}

    __device__ virtual ~constant_medium() {
        if (boundary) {
            delete boundary;
            boundary = nullptr;
        }
        if (phase_function) {
            delete phase_function;
            phase_function = nullptr;
        }
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;
    
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const {
        return boundary->bounding_box(time0, time1, output_box);
    };

public:
    hittable* boundary;
    material* phase_function;
    float neg_inv_density;
};

__device__ bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    const bool enableDebug = false;
    const bool debugging = enableDebug && random_float(rng) < 0.00001;

    hit_record rec1, rec2;

    if (!boundary->hit(r, -infinity, infinity, rec1))
        return false;

    if (!boundary->hit(r, rec1.t + 0.0001f, infinity, rec2))
        return false;

    // TODO: make printing for device code
    if (debugging)
        printf("\nt_min=%f, t_max=%f\n", rec1.t, rec2.t);
//        std::cerr << "\nt_min=" << rec1.t << ", t_max=" << rec2.t << '\n';

    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = glm::length(r.direction());
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * glm::log(random_float(rng));

    // Check if hit ray passed completely through the medium
    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    // TODO: see above
    if (debugging) {
        printf("hit_distance = %f\nrec.t = %f\nrec.p = (%f, %f, %f)\n", hit_distance, rec.t, 
            rec.p.x, rec.p.y, rec.p.z);
        //std::cerr << "hit_distance = " << hit_distance << '\n'
        //          << "rec.t = " << rec.t << '\n'
        //          << "rec.p = " << rec.p << '\n';
    }
    rec.normal = glm::vec3(1, 0, 0);
    rec.front_face = true;
    rec.mat_ptr = phase_function;

    return true;
}