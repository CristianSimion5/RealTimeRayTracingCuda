#pragma once

#include <iostream>
#include <algorithm>

#include "utility.h"

#include "hittable.h"
#include "hittable_list.h"

class bvh_node : public hittable {
public:
    __device__ bvh_node() = default;

    __device__ bvh_node(const hittable_list& list, float time0, float time1, curandState* rand_state)
        : bvh_node(list.objects, 0, list.objects_size, time0, time1, rand_state)
    {}

    __device__ ~bvh_node() {
        if (left)  delete left;
        if (right) delete right;
    }

    __device__ bvh_node(
        hittable** const src_objects,
        int start, int end, float time0, float time1, curandState* rand_state);

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng = nullptr) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
    hittable* left  = nullptr;
    hittable* right = nullptr;
    aabb box;
};

__device__ inline bool box_compare(const hittable* a, const hittable* b, int axis) {
    aabb box_a;
    aabb box_b;
    
    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b)) {
        //std::cerr << "No bounding box in bvh_node constructor.\n";
        printf("No bounding box in bvh_node constructor.\n");
    }
    

    return box_a.min()[axis] < box_b.min()[axis];
}

__device__ bool box_x_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 0);
}

__device__ bool box_y_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 1);
}

__device__ bool box_z_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 2);
}

__device__ void insertion_sort(hittable** const l, int beg, int end, 
    bool (*comparator)(const hittable* a, const hittable* b)) {
    for (int i = beg + 1; i < end; i++) {
        auto current = l[i];
        int j = i - 1;
        while (j >= beg && comparator(l[j], current)) {
            l[j + 1] = l[j];
            j--;
        }
        l[j + 1] = current;
    }

    for (int i = beg + 1; i < end; i++) {
        if (comparator(l[i], l[i - 1])) {
            printf("ok");
        }
    }
}

__device__ bvh_node::bvh_node(
    hittable** const src_objects, 
    int start, int end, float time0, float time1, curandState* rand_state
) {
    //hittable** objects = new hittable*[end - start];
    //for (int i = 0)
    auto objects = src_objects;

    int axis = random_int(2, rand_state);
    printf("axis: %d\n", axis);
    auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            left = objects[start + 1];
            right = objects[start];
        }
    } else {
        //insertion_sort(objects, start, end, comparator);
        for (int i = start + 1; i < end; i++) {
            auto current = objects[i];
            int j = i - 1;
            while (j >= start && comparator(objects[j], current)) {
                objects[j + 1] = objects[j];

                j--;
            }
            objects[j + 1] = current;
        }

        auto mid = start + object_span / 2;
        left = new bvh_node(objects, start, mid, time0, time1, rand_state);
        right = new bvh_node(objects, mid, end, time0, time1, rand_state);
    }

    aabb box_left, box_right;
    if (!left->bounding_box(time0, time1, box_left) || 
        !right->bounding_box(time0, time1, box_right)) {
        //std::cerr << "No bounding box in bvh_node constructor.\n";
        printf("No bounding box in bvh_node constructor.\n");
    }

    box = surrounding_box(box_left, box_right);
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* rng) const {
    if (!box.hit(r, t_min, t_max)) 
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ bool bvh_node::bounding_box(float time0, float time1, aabb& output_box) const {
    output_box = box;
    return true;
}