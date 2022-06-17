#pragma once

#include "utility.h"

#include "hittable.h"
#include "material.h"

class xy_rect : public hittable {
public:
    __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {}
    
    __device__ virtual ~xy_rect() { 
        if (mp) {
            delete mp;
            mp = nullptr;
        }
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        output_box = aabb(point3(x0, y0, k - 1e-4f), point3(x0, y0, k + 1e-4f));
        return true;
    }
    
public:
    material* mp;
    float x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    if (glm::abs(r.direction().z) < 1e-4f)
        return false;

    auto t = (k - r.origin().z) / r.direction().z;
    if (t < t_min || t_max < t)
        return false;

    auto x = r.origin().x + t * r.direction().x;
    auto y = r.origin().y + t * r.direction().y;
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;

    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;

    auto outward_normal = glm::vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);

    return true;
}

class xz_rect : public hittable {
public:
    __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

    __device__ virtual ~xz_rect() {
        if (mp) {
            delete mp;
            mp = nullptr;
        }
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        output_box = aabb(point3(x0, k - 1e-4f, z0), point3(x0, k + 1e-4f, z1));
        return true;
    }

    __device__ virtual float pdf_value(const point3& origin, const glm::vec3& dir) const override;
    __device__ virtual glm::vec3 random_surface_point(const point3& o, curandState* rand, color& emittance) const override;

public:
    material* mp;
    float x0, x1, z0, z1, k;
};

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    if (glm::abs(r.direction().y) < 1e-4f)
        return false;

    auto t = (k - r.origin().y) / r.direction().y;
    if (t < t_min || t_max < t)
        return false;

    auto x = r.origin().x + t * r.direction().x;
    auto z = r.origin().z + t * r.direction().z;
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;

    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;

    auto outward_normal = glm::vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);

    return true;
}

inline __device__ float xz_rect::pdf_value(const point3& origin, const glm::vec3& dir) const {
    hit_record rec;
    if (!hit(ray(origin, dir), 0.001f, infinity, rec))
        return 0.0f;

    auto area = (x1 - x0) * (z1 - z0);
    auto dist2 = rec.t * rec.t * glm::length2(dir);
    auto cosine = glm::abs(glm::dot(dir, rec.normal) / glm::length(dir));

    return dist2 / (cosine * area + 1e-4f);
}

__device__ glm::vec3 xz_rect::random_surface_point(const point3& o, curandState* rand, color& emittance) const {
    float u = random_float(rand);
    float v = random_float(rand);
    point3 p(u * (x1 - x0) + x0, k, v * (z1 - z0) + z0);
    hit_record temp_rec;
    temp_rec.front_face = false;
    emittance = mp->emitted(ray(), temp_rec, u, v, p);

    return p - o;
}

class yz_rect : public hittable {
public:
    __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

    __device__ virtual ~yz_rect() {
        if (mp) {
            delete mp;
            mp = nullptr;
        }
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        output_box = aabb(point3(k - 1e-4f, y0, z0), point3(k + 1e-4f, y1, z1));
        return true;
    }

public:
    material* mp;
    float y0, y1, z0, z1, k;
};

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    if (glm::abs(r.direction().x) < 1e-4f)
        return false;

    auto t = (k - r.origin().x) / r.direction().x;
    if (t < t_min || t_max < t)
        return false;

    auto y = r.origin().y + t * r.direction().y;
    auto z = r.origin().z + t * r.direction().z;
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;

    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;

    auto outward_normal = glm::vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);

    return true;
}