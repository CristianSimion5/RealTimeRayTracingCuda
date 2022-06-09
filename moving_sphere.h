#pragma once

#include "utility.h"

#include "hittable.h"
#include "aabb.h"

// NOTE: any material sent to the hittable should be managed from the hittable from then on
class moving_sphere : public hittable {
public:
    __device__ moving_sphere() = default;
    __device__ moving_sphere(
        point3 cen0, point3 cen1, float _time0, float _time1, float r, material* m) 
        : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m)
    {};
    __device__ virtual ~moving_sphere() { 
        if (mat_ptr) {
            delete mat_ptr;
            mat_ptr = nullptr;
        }
    }

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float _time0, float _time1, aabb& output_box) const override;

    __device__ point3 center(float time) const;

public:
    point3 center0, center1;
    float time0, time1;
    float radius;
    material* mat_ptr;
};

__device__ point3 moving_sphere::center(float time) const {
    float t_int = (time - time0) / (time1 - time0);
    return center0 +  t_int * (center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    glm::vec3 oc = r.origin() - center(r.time());
    auto a = glm::length2(r.direction());
    auto half_b = glm::dot(r.direction(), oc);
    auto c = glm::length2(oc) - radius * radius;
    auto det = half_b * half_b - a * c;
    if (det < 0) {
        return false;
    }

    auto sqrt_det = glm::sqrt(det);
    auto root = (-half_b - sqrt_det) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_det) / a;
        if (root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    glm::vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ bool moving_sphere::bounding_box(float _time0, float _time1, aabb& output_box) const {
    aabb box0(
        center(_time0) - glm::vec3(radius),
        center(_time0) + glm::vec3(radius));
    aabb box1(
        center(_time1) - glm::vec3(radius),
        center(_time1) + glm::vec3(radius));
    output_box = surrounding_box(box0, box1);
    return true;
}