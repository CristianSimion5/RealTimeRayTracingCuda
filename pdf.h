#pragma once

#include "utility.h"
#include "vector3.h"
#include "onb.h"
#include "hittable.h"

class pdf {
public:
    __device__ virtual ~pdf() {}

    __device__ virtual float value(const glm::vec3& direction) const = 0;
    __device__ virtual glm::vec3 generate(curandState* rand_state, color& emittance) const = 0;
};

class cosine_pdf : public pdf {
public:
    __device__ cosine_pdf() {}
    __device__ cosine_pdf(const glm::vec3& normal) { uvw.build_from_norm(normal); }

    __device__ virtual float value(const glm::vec3& direction) const override {
        auto cosine = glm::dot(glm::normalize(direction), uvw.w());
        return cosine < 0.0f ? 0.0f : cosine / pi;
    }

    __device__ virtual glm::vec3 generate(curandState* rand_state, color& emittance) const override {
        return uvw.local(random_cosine_direction(rand_state));
    }

public:
    onb uvw;
};

class hittable_pdf : public pdf {
public:
    __device__ hittable_pdf(hittable* p, const point3& origin) : ptr(p), o(origin) {}

    __device__ virtual float value(const glm::vec3& direction) const override {
        return ptr->pdf_value(o, direction);
    }
    __device__ virtual glm::vec3 generate(curandState* rand_state, color& emittance) const override {
        return ptr->random_surface_point(o, rand_state, emittance);
    }

public:
    point3 o;
    hittable* ptr;
};

class mixture_pdf : public pdf {
public:
    __device__ mixture_pdf(pdf* p1, pdf* p2) : p{ p1, p2 } {}
    
    __device__ virtual float value(const glm::vec3& direction) const override {
        return 0.5f * p[0]->value(direction) + 0.5f * p[1]->value(direction);
    }
    __device__ virtual glm::vec3 generate(curandState* rand_state, color& emitted) const override {
        if (random_float(rand_state) < 0.5f)
            return p[0]->generate(rand_state, emitted);
        else
            return p[1]->generate(rand_state, emitted);
    }

public:
    pdf* p[2];
};