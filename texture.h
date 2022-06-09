#pragma once

#include "stb/stb_image_implementation.h"

#include "utility.h"

#include "perlin.h"

class _texture {
public:
    __device__ virtual ~_texture() {}
    __device__ virtual color value(float u, float v, const point3& p) const = 0;
};

class solid_color : public _texture {
public:
    __device__ solid_color(color c) : color_value(c) {}

    __device__ solid_color(float red, float green, float blue)
        : color_value(red, green, blue) {}

    __device__ virtual color value(float u, float v, const point3& p) const override {
        return color_value;
    }

public:
    color color_value;
};

class checker_texture : public _texture {
public:
    __device__ checker_texture(_texture* _even, _texture* _odd)
        : even(_even), odd(_odd) {}

    __device__ checker_texture(color c1, color c2)
        : even(new solid_color(c1)), odd(new solid_color(c2)) {}

    __device__ virtual ~checker_texture() {
        printf("Called checker texture destructor\n");
        delete odd;
        delete even;
    }

    __device__ virtual color value(float u, float v, const point3& p) const override {
        auto sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

public:
    _texture* odd;
    _texture* even;
};

class noise_texture : public _texture {
public:
    __device__ noise_texture(float sc, curandState* rand_state) : scale(sc), noise(rand_state) {}

    __device__ virtual color value(float u, float v, const point3& p) const override {
        return color(1, 1, 1) * 0.5f * (1.f + sin(scale * p.z + 10 * noise.turb(p)));
            //* noise.turb(scale * p); // 0.5f * (1.f + noise.noise(scale * p));
    }

public:
    perlin noise;
    float scale;
};

/*
class image {
public:
    const static int bytes_per_pixel = 3;
    
    __host__ image() : d_data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

    __host__ image(const char* filename) {
        auto components_per_pixel = bytes_per_pixel;

        auto h_data = stbi_load(
            filename, &width, &height, &components_per_pixel, components_per_pixel);

        if (!h_data) {
            //std::cerr << "ERROR: Could not load texture image file: \"" << filename << "\".\n";
            printf("ERROR: Could not load texture image file: \"%s\".\n", filename);
            width = height = 0;
        }
        else {
            assert(cudaMallocManaged(&d_data, width * height * components_per_pixel * sizeof(unsigned char)) == cudaSuccess);
            memcpy(d_data, h_data, width * height * components_per_pixel * sizeof(unsigned char));
        }

        bytes_per_scanline = bytes_per_pixel * width;
        stbi_image_free(h_data);
    }

public:
    unsigned char* d_data;  // Image data is stored only on device (GPU)
    int width, height;
    int bytes_per_scanline;
};
*/

struct image_params {
    int width, height;
    int bytes_per_scanline;
};

__host__ void load_texture_data(const char* filename, unsigned char**& texture_buffer, image_params*& texture_info, int tex_index) {
    int components_per_pixel = 3, width, height;

    unsigned char* h_data = stbi_load(
        filename, &width, &height, &components_per_pixel, components_per_pixel);
    texture_buffer[tex_index] = nullptr;

    if (!h_data) {
        //std::cerr << "ERROR: Could not load texture image file: \"" << filename << "\".\n";
        printf("ERROR: Could not load texture image file: \"%s\".\n", filename);
        width = height = 0;
    }
    else {
        cudaMallocManaged(&texture_buffer[tex_index], 
            width * height * components_per_pixel * sizeof(unsigned char));
        memcpy(texture_buffer[tex_index], h_data, width * height * components_per_pixel);
    }
    texture_info[tex_index].width = width;
    texture_info[tex_index].height = height;
    texture_info[tex_index].bytes_per_scanline = width * components_per_pixel;

    stbi_image_free(h_data);
}


class image_texture : public _texture {
public:
    const static int bytes_per_pixel = 3;

    __device__ image_texture() : d_data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

    __device__ image_texture(unsigned char* image_data, int w, int h, int bps) : d_data(image_data), width(w), height(h), bytes_per_scanline(bps) {}

    __device__ image_texture(unsigned char* image_data, image_params image_info) : d_data(image_data), 
        width(image_info.width), height(image_info.height), bytes_per_scanline(image_info.bytes_per_scanline) {}
    
    //__device__ image_texture(const image& img) : image_texture(img.d_data, img.width, img.height, img.bytes_per_scanline) {}

    /*
    __host__ image_texture(const image_texture& other) {
        printf("Called image texture copy constructor\n");
        
        width = other.width;
        height = other.height;
        bytes_per_scanline = other.bytes_per_scanline;

        assert(cudaMalloc(&d_data, width * height * bytes_per_pixel * sizeof(unsigned char)) == cudaSuccess);
        assert(cudaMemcpy(d_data, other.d_data,
            width * height * bytes_per_pixel * sizeof(unsigned char), cudaMemcpyDeviceToDevice) == cudaSuccess);
    }

    __host__ image_texture& operator=(const image_texture& other) {
        if (this != &other) {
            printf("Called image texture assignment\n");
            if (d_data)
                cudaFree(d_data);

            width = other.width;
            height = other.height;
            bytes_per_scanline = other.bytes_per_scanline;

            assert(cudaMalloc(&d_data, width * height * bytes_per_pixel * sizeof(unsigned char)) == cudaSuccess);
            assert(cudaMemcpy(d_data, other.d_data,
                width * height * bytes_per_pixel * sizeof(unsigned char), cudaMemcpyDeviceToDevice) == cudaSuccess);
        }
        return *this;
    }
    */

    __device__ virtual color value(float u, float v, const point3& p) const override {
        if (d_data == nullptr)
            return color(0, 1, 1);

        u = clamp(u, 0.0f, 1.0f);
        v = 1.0f - clamp(v, 0.0f, 1.0f);

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);
        
        if (i >= width)  i = width - 1;
        if (j >= height) j = height - 1;

        const auto color_scale = 1.0f / 255.0f;
        auto pixel = d_data + j * bytes_per_scanline + i * bytes_per_pixel;

        return color(color_scale * pixel[0], 
                     color_scale * pixel[1], 
                     color_scale * pixel[2]);
    }

public:
    unsigned char* d_data;  // Image data is stored only on device (GPU)
    int width, height;
    int bytes_per_scanline;
};