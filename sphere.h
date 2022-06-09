#pragma once

#include "hittable.h"
#include "vector3.h"

// NOTE: any material sent to the hittable should be managed from the hittable from then on
class sphere: public hittable {
public:
    __device__ sphere() = default;
    __device__ sphere(point3 cen, float r, material *m)
        : center(cen), radius(r), mat_ptr(m) {};
    __device__ virtual ~sphere() { 
        if (mat_ptr) {
            delete mat_ptr;
            mat_ptr = nullptr;
        }
    }

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
    point3 center;
    float radius;
    material* mat_ptr;

private:
    __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
        auto theta = acos(-p.y);
        auto phi = atan2(-p.z, p.x) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    glm::vec3 oc = r.origin() - center;
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
    glm::vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;

    return true;
}

bool sphere::bounding_box(float time0, float time1, aabb& output_box) const {
    output_box = aabb(
        center - glm::vec3(radius),
        center + glm::vec3(radius)
    );
    return true;
}