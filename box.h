#pragma once

#include "utility.h"

#include "aarect.h"
#include "hittable_list.h"

class box : public hittable {
public:
    __device__ box(const point3& p0, const point3& p1, material* ptr);

    __device__ virtual ~box() {
        for (int i = 0; i < 6; i++)
            delete sides.objects[i];
        delete[] sides.objects;
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        output_box = aabb(box_min, box_max);
        return true;
    }

public:
    point3 box_min;
    point3 box_max;
    hittable_list sides;
};

__device__ box::box(const point3& p0, const point3& p1, material* ptr) {
    box_min = p0;
    box_max = p1;

    sides.objects_size = 6;
    sides.objects = new hittable*[6];

    sides.objects[0] = new xy_rect(p0.x, p1.x, p0.y, p1.y, p1.z, ptr);
    sides.objects[1] = new xy_rect(p0.x, p1.x, p0.y, p1.y, p0.z, ptr);

    sides.objects[2] = new xz_rect(p0.x, p1.x, p0.z, p1.z, p1.y, ptr);
    sides.objects[3] = new xz_rect(p0.x, p1.x, p0.z, p1.z, p0.y, ptr);

    sides.objects[4] = new yz_rect(p0.y, p1.y, p0.z, p1.z, p1.x, ptr);
    sides.objects[5] = new yz_rect(p0.y, p1.y, p0.z, p1.z, p0.x, ptr);

    sides.update_box(0.0f, 1.0f);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    bool result = sides.hit(r, t_min, t_max, rec);
    return result;
}