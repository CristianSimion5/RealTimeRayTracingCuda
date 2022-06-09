#pragma once

#include "utility.h"

class camera {
public:
    __device__ camera(
        point3    orig,
        point3    lookat,
        glm::vec3 vup,
        float     vfov,
        float     aspect_ratio,
        float     aperture,
        float     focus_dist,
        float     _time0 = 0.0f,
        float     _time1 = 0.0f
    ) {
        auto theta = radians(vfov);
        auto h = glm::tan(theta / 2);
        auto viewport_height = 2.0f * h;
        auto viewport_width = aspect_ratio * viewport_height;
       
        origin = orig;
        w = glm::normalize(origin - lookat);
        u = glm::normalize(glm::cross(vup, w));
        v = glm::cross(w, u);

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f -  focus_dist * w;

        lens_radius = aperture / 2;
        time0 = _time0;
        time1 = _time1;
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


private:
    point3 origin;
    point3 lower_left_corner;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 u, v, w;
    float lens_radius;
    float time0, time1;
};

