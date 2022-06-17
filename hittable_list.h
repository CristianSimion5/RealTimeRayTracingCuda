#pragma once

#include "hittable.h"

#include "aabb.h"

class hittable_list: public hittable {
public:
    __device__ hittable_list() = default;
    __device__ hittable_list(hittable** l, int n) { 
        objects = l; 
        objects_size = n;
        bounding_box(0.0f, 1.0f, box);
    }

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(
        float time0, float time1, aabb& output_box) const override;

    __device__ void update_box(float time0, float time1);

    __device__ bool any_hit(const ray& r, float t_min, float t_max, curandState* rng = nullptr);

    __device__ virtual float pdf_value(const point3& o, const glm::vec3& dir) const override;

    __device__ virtual glm::vec3 random_surface_point(const point3& o, curandState* rand,
        color& emittance) const override;

public:
    hittable** objects;
    int objects_size;
    aabb box;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    if (!box.hit(r, t_min, t_max)) return false;
    
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < objects_size; i++) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec, rng)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
    if (objects_size == 0) return false;

    aabb temp_box;
    bool first_flag = true;

    for (int i = 0; i < objects_size; i++) {
        if (!objects[i]->bounding_box(time0, time1, temp_box)) 
            return false;
        output_box = first_flag ? temp_box : surrounding_box(output_box, temp_box);
        first_flag = false;
    }

    return true;
}

inline  __device__ void hittable_list::update_box(float time0, float time1) {
    bounding_box(time0, time1, box);
}

__device__ bool hittable_list::any_hit(const ray& r, float t_min, float t_max, curandState* rng) {
    if (!box.hit(r, t_min, t_max)) return false;

    hit_record dummy_rec;

    for (int i = 0; i < objects_size; i++) {
        if (objects[i]->hit(r, t_min, t_max, dummy_rec, rng)) {
            return true;
        }
    }

    return false;
}

__device__ float hittable_list::pdf_value(const point3& o, const glm::vec3& dir) const {
    auto weight = 1.0f / objects_size;
    auto sum = 0.0f;

    for (int i = 0; i < objects_size; i++) {
        sum += weight * objects[i]->pdf_value(o, dir);
    }

    return sum;
}

__device__ glm::vec3 hittable_list::random_surface_point(const point3& o, curandState* rand, color& emittance) const {
    return objects[random_int(objects_size - 1, rand)]->random_surface_point(o, rand, emittance);
}
