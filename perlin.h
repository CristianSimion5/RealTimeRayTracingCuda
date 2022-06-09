#pragma once

#include "utility.h"
#include "vector3.h"

class perlin {
public:
    __device__ perlin(curandState* rand_state) {
        ranvec = new glm::vec3[point_count];
        for (int i = 0; i < point_count; i++) {
            ranvec[i] = glm::normalize(random(-1.f, 1.f, rand_state));
        }

        perm_x = perlin_generate_perm(rand_state);
        perm_y = perlin_generate_perm(rand_state);
        perm_z = perlin_generate_perm(rand_state);
    }

    __device__ ~perlin() {
        delete[] ranvec;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
    }

    __device__ float noise(const point3& p) const {
        auto u = p.x - floor(p.x);
        auto v = p.y - floor(p.y);
        auto w = p.z - floor(p.z);

        auto i = static_cast<int>(floor(p.x));
        auto j = static_cast<int>(floor(p.y));
        auto k = static_cast<int>(floor(p.z));
        glm::vec3 c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++) {
                    c[di][dj][dk] = ranvec[
                        perm_x[(i + di) & 255] ^
                        perm_y[(j + dj) & 255] ^
                        perm_z[(k + dk) & 255]
                    ];
                }

        return perlin_interp(c, u, v, w);
    }

    __device__ float turb(const point3& p, int depth = 7) const {
        auto accum = 0.f;
        auto temp_p = p;
        auto weight = 1.f;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5f;
            temp_p *= 2;
        }

        return abs(accum);
    }

private:
    static const int point_count = 256;
    glm::vec3* ranvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    __device__ static int* perlin_generate_perm(curandState* rand_state) {
        auto p = new int[point_count];
        for (int i = 0; i < perlin::point_count; i++) {
            p[i] = i;
        }

        permute(p, point_count, rand_state);

        return p;
    }

    __device__ static void permute(int* p, int n, curandState* rand_state) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(i, rand_state);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __device__ static float trilinear_interp(float c[2][2][2], float u, float v, float w) {
        auto accum = 0.f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    accum += (i * u + (1 - i) * (1 - u)) *
                             (j * v + (1 - j) * (1 - v)) *
                             (k * w + (1 - k) * (1 - w)) *
                             c[i][j][k];
                }
        return accum;
    }

    __device__ static float perlin_interp(glm::vec3 c[2][2][2], float u, float v, float w) {
        auto accum = 0.f;
        auto uu = u * u * (3 - 2 * u);
        auto vv = v * v * (3 - 2 * v);
        auto ww = w * w * (3 - 2 * w);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    glm::vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu)) *
                             (j * vv + (1 - j) * (1 - vv)) *
                             (k * ww + (1 - k) * (1 - ww)) *
                             dot(c[i][j][k], weight_v);
                }
        return accum;
    }
};

