#pragma once

#include "utility.h"

class camera {
public:
    __host__ __device__ camera(
        point3    orig,
        point3    lookat,
        glm::vec3 vup,
        float     vfov,
        float     aspect_ratio,
        float     aperture,
        float     focus_dist,
        float     _time0 = 0.0f,
        float     _time1 = 0.0f
    ) : origin(orig), lookat(lookat), vup(vup), vfov(vfov), aspect(aspect_ratio), 
        aperture(aperture), focus_dist(focus_dist), time0(_time0), time1(_time1) {
        update_camera();
    }

    __device__ ray get_ray(float s, float t, curandState* rand_state) const {
        glm::vec3 rd = lens_radius * random_in_unit_disk(rand_state);
        glm::vec3 offset = u * rd.x + v * rd.y;
        
        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            random_float(time0, time1, rand_state)
        );
    }

    __host__ __device__ void update_camera() {
        auto theta = radians(vfov);
        auto h = glm::tan(theta / 2);
        auto viewport_height = 2.0f * h;
        auto viewport_width = aspect * viewport_height;

        w = glm::normalize(origin - lookat);
        u = glm::normalize(glm::cross(vup, w));
        v = glm::cross(w, u);

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;

        lens_radius = aperture / 2;
    }

    __host__ void move_camera(const glm::vec3& offset) {
        origin += offset;
    }

public:
    point3 origin;
    point3 lookat;
    glm::vec3 vup;
    float aspect;
    float vfov;
    float aperture;
    float focus_dist;
    float time0, time1;
private:
    point3 lower_left_corner;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 u, v, w;
    float lens_radius;
};

