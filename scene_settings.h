#pragma once

#include "utility.h"
#include "cuda_utils.h"
#include "texture.h"

#include "hittable.h"
#include "hittable_list.h"
#include "camera.h"

class scene_settings {
public:
    static const int MAX_WIDTH = 3840;
    static const int MAX_HEIGHT = 2160;
    static const int MAX_PIXELS = MAX_WIDTH * MAX_HEIGHT;

    scene_settings(int nx, int ny, int spp, int max_depth, color bgcolor, 
        int num_objects, int num_textures)
        : width(nx), height(ny), spp(spp), max_bounces(max_depth), bgcolor(bgcolor), 
        num_objects(num_objects), num_textures(num_textures) {
        float aspect_ratio = 1.0f * width / height;

        // Initialize cuda random per thread
        checkCudaErrors(cudaMalloc((void**)&d_rand_state, MAX_PIXELS * sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void**)&d_rand_state_world_gen, sizeof(curandState)));
        rand_init << <1, 1 >> > (d_rand_state_world_gen);
    
        // Initialize textures
        checkCudaErrors(cudaMallocManaged((void**)&image_textures, num_textures * sizeof(unsigned char*)));
        checkCudaErrors(cudaMallocManaged(&image_info, num_textures * sizeof(image_params)));

        load_texture_data("textures/earthmap.jpg", image_textures, image_info, 0);
        load_texture_data("textures/happyface.jpg", image_textures, image_info, 1);

        checkCudaErrors(cudaMalloc((void**)&d_list, num_objects * sizeof(hittable*)));
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable_list*)));
        
        checkCudaErrors(cudaMallocManaged((void**)&d_camera, sizeof(camera*)));
        checkCudaErrors(cudaMallocManaged(d_camera, sizeof(camera)));

        create_world << <1, 1 >> > (d_list, num_objects, d_world, d_camera, 
            image_textures, image_info, aspect_ratio, d_rand_state_world_gen);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        dim3 blocks(MAX_WIDTH / threads.x + 1, MAX_HEIGHT / threads.y + 1);
        
        render_init << <blocks, threads >> > (MAX_WIDTH, MAX_HEIGHT, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        continuous_frame_count = 1;
    }

    ~scene_settings() {
        checkCudaErrors(cudaDeviceSynchronize());
        free_world << <1, 1 >> > (d_list, num_objects, d_world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(image_info));
        checkCudaErrors(cudaFree(image_textures));
        checkCudaErrors(cudaFree(d_camera));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(d_list));
        cudaDeviceReset();
    }

    void generate_frame(color* fb, color* pbo_buffer) {
        dim3 blocks = dim3(width / threads.x + 1, height / threads.y + 1);
        render<<<blocks, threads>>>(fb, pbo_buffer, width, height, spp, max_bounces,
            d_camera, bgcolor, d_world, d_rand_state, continuous_frame_count);
        continuous_frame_count++;
    }

    void move_camera(const glm::vec3& offset) {
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        (**d_camera).move_camera(offset);
        (**d_camera).update_camera();
        continuous_frame_count = 1;
    }

public:
    int width, height, spp, max_bounces;
    color bgcolor;
    int num_objects, num_textures;
    const dim3 threads = dim3(16, 16);

    unsigned char** image_textures;
    image_params* image_info;

    curandState* d_rand_state;
    curandState* d_rand_state_world_gen;

    hittable** d_list;
    hittable_list** d_world;
    camera** d_camera;
    int continuous_frame_count;
};

