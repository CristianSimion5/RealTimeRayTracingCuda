#pragma once

#include <iostream>

#include "utility.h"
#include "hittable.h"
#include "texture.h"

class material {
public:
    __device__ virtual ~material() {}
    
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, 
        curandState* rand_state
    ) const = 0;

    __device__ virtual color emitted(float u, float v, const point3& p) const {
        return color(0.0f);
    }
};

// NOTE: any texture sent to the material should be managed from the material from then on
class lambertian : public material {
public:
    __device__ lambertian(const color& a) : albedo(new solid_color(a)) {}
    __device__ lambertian(_texture* a) : albedo(a) {}
    __device__ virtual ~lambertian() { delete albedo; }

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, 
        curandState* rand_state
    ) const override {
        // TODO: use random in hemisphere
        auto scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Degenerate direction
        if (near_zero(scatter_direction))
        {
            /*std::cerr << scatter_direction.x << ' ' << scatter_direction.y <<
                ' ' << scatter_direction.z << '\n';*/
            scatter_direction = rec.normal;
        }

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo->value(rec.u, rec.v,  rec.p);
        return true;
    }

public:
    _texture* albedo;
};

class metal : public material {
public:
    __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, 
        curandState* rand_state
    ) const override {
        glm::vec3 reflected = glm::reflect(glm::normalize(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand_state), r_in.time());
        attenuation = albedo;
        return (glm::dot(scattered.direction(), rec.normal) > 0);
    }

public:
    color albedo;
    float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
        curandState* rand_state
    ) const override {
        attenuation = color(1.0f);
        float refraction_ratio = rec.front_face ? (1.f / ir) : ir;

        glm::vec3 unit_direction = glm::normalize(r_in.direction());
        float cos_theta = std::fmin(glm::dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        glm::vec3 direction;
        
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(rand_state))
            direction = glm::reflect(unit_direction, rec.normal);
        else
            direction = glm::refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }

public:
    float ir;

private:
    __device__ static float reflectance(float cosine, float ref_idx) {
        // Schlick's approximation
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.f - r0) * pow((1.f - cosine), 5);
    }
};

class diffuse_light : public material {
public:
    __device__ diffuse_light(_texture* a) : emit(a) {}
    __device__ diffuse_light(color c) : emit(new solid_color(c)) {}
    __device__ virtual ~diffuse_light() {
        if (emit) {
            delete emit;
            emit = nullptr;
        }
    }

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
        curandState* rand_state
    ) const override {
        return false;
    }

    __device__ virtual color emitted(float u, float v, const point3& p) const override {
        return emit->value(u, v, p);
    }

public:
    _texture* emit;
};

class isotropic : public material {
public:
    __device__ isotropic(color c) : albedo(new solid_color(c)) {}
    __device__ isotropic(_texture* a) : albedo(a) {}
    __device__ virtual ~isotropic() {
        if (albedo) {
            delete albedo;
            albedo = nullptr;
        }
    }

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
        curandState* rand_state
    ) const override {
        scattered = ray(rec.p, random_in_unit_sphere(rand_state), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    };

public:
    _texture* albedo;
};