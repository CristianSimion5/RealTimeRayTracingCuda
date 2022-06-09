#pragma once

#include "ray.h"
#include "utility.h"
#include "aabb.h"

class material;

struct hit_record {
    point3 p;
    glm::vec3 normal;
    material *mat_ptr;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    __device__ ~hittable() {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const = 0;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
};

class translate : public hittable {
public:
    __device__ translate(hittable* p, const glm::vec3& displacement)
        : ptr(p), offset(displacement) {}
    __device__ virtual ~translate() {
        if (ptr) {
            delete ptr;
            ptr = nullptr;
        }
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
    hittable* ptr;
    glm::vec3 offset;
};

__device__ bool translate::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    ray moved_r(r.origin() - offset, r.direction(), r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec))
        return false;

    rec.p += offset;
    rec.set_face_normal(moved_r, rec.normal);

    return true;
}

__device__ bool translate::bounding_box(float time0, float time1, aabb& output_box) const {
    if (!ptr->bounding_box(time0, time1, output_box))
        return false;

    output_box = aabb(
        output_box.min() + offset,
        output_box.max() + offset
    );

    return true;
}

class rotate_y : public hittable {
public:
    __device__ rotate_y(hittable* p, float angle);
    __device__ virtual ~rotate_y() {
        if (ptr) {
            delete ptr;
            ptr = nullptr;
        }
    }
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        output_box = bbox;
        return has_box;
    };

public:
    hittable* ptr;
    float sin_theta;
    float cos_theta;
    bool has_box;
    aabb bbox;
};

__device__ rotate_y::rotate_y(hittable* p, float angle)
    : ptr(p) {
    auto rads = radians(angle);
    sin_theta = sin(rads);
    cos_theta = cos(rads);
    has_box = ptr->bounding_box(0, 1, bbox);

    point3 min( infinity);
    point3 max(-infinity);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i * bbox.max().x - (1 - i) * bbox.min().x;
                auto y = j * bbox.max().y - (1 - j) * bbox.min().y;
                auto z = k * bbox.max().z - (1 - k) * bbox.min().z;
            
                auto newx =  cos_theta * x + sin_theta * z;
                auto newz = -sin_theta * x + cos_theta * z;

                glm::vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}

__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin.x = cos_theta * r.origin().x - sin_theta * r.origin().z;
    origin.z = sin_theta * r.origin().x + cos_theta * r.origin().z;

    direction.x = cos_theta * r.direction().x - sin_theta * r.direction().z;
    direction.z = sin_theta * r.direction().x + cos_theta * r.direction().z;

    ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p.x =  cos_theta * rec.p.x + sin_theta * rec.p.z;
    p.z = -sin_theta * rec.p.x + cos_theta * rec.p.z;

    normal.x =  cos_theta * rec.normal.x + sin_theta * rec.normal.z;
    normal.z = -sin_theta * rec.normal.x + cos_theta * rec.normal.z;

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}