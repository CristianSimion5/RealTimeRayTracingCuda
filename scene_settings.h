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

        checkCudaErrors(cudaMallocManaged(&offset, 25 * sizeof(glm::ivec2)));
        checkCudaErrors(cudaMallocManaged(&kernel, 25 * sizeof(float)));
        float h[5]{ 1.f / 16, 1.f / 4, 3.f / 8, 1.f / 4, 1.f / 16 };
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                offset[i * 5 + j] = glm::ivec2(j - 2, i - 2);
                kernel[i * 5 + j] = h[i] * h[j];
            }
        }

        checkCudaErrors(cudaMalloc(&temp_buf, MAX_PIXELS * sizeof(color)));

        denoise_passes = 0;
    }

    ~scene_settings() {
        checkCudaErrors(cudaDeviceSynchronize());
        free_world << <1, 1 >> > (d_list, num_objects, d_world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(temp_buf));
        checkCudaErrors(cudaFree(offset));
        checkCudaErrors(cudaFree(kernel));
        checkCudaErrors(cudaFree(image_info));
        checkCudaErrors(cudaFree(image_textures));
        checkCudaErrors(cudaFree(d_camera));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(d_list));
        cudaDeviceReset();
    }

    void generate_frame(color* fb, glm::vec3* normals, point3* positions, color* pbo_buffer) {
        dim3 blocks = dim3(width / threads.x + 1, height / threads.y + 1);
        render<<<blocks, threads>>>(fb, pbo_buffer, normals, positions, width, height, spp, max_bounces,
            d_camera, bgcolor, d_world, d_rand_state, continuous_frame_count);

        continuous_frame_count++;

        if (denoise_passes < 1) return;

        // Wait for the frame to generate
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Denoise
        int len = width * height;
        float c_phi = 0.0f, n_phi, p_phi;
        float norm_sum = 0.0f, pos_sum = 0.0f;
        for (int i = 0; i < len; i++) {
            c_phi = glm::max(c_phi, glm::max(fb[i].r, 
                glm::max(fb[i].g, fb[i].b)));

            norm_sum += normals[i].x + normals[i].y + normals[i].z;
            pos_sum += positions[i].x + positions[i].y + positions[i].z;
        }
        float norm_mean = norm_sum / len * 3;
        float pos_mean = pos_sum / len * 3;
        norm_sum = pos_sum = 0.0f;

        for (int i = 0; i < len; i++) {
            auto dif = normals[i] - norm_mean;
            dif = dif * dif;
            norm_sum += normals[i].x + normals[i].y + normals[i].z;

            dif = positions[i] - pos_mean;
            dif = dif * dif;
            pos_sum += positions[i].x + positions[i].y + positions[i].z;
        }

        c_phi = c_phi / (1.f * (continuous_frame_count - 1) * spp);
        n_phi = norm_sum / len * 3;
        p_phi = pos_sum / len * 3;;
        int stepwidth = 1;

        //std::cout << c_phi << ' ' << n_phi << ' ' << p_phi << '\n';
        //checkCudaErrors(cudaMemcpy(temp_buf, pbo_buffer, width * height * sizeof(color), cudaMemcpyDeviceToDevice));
        
        for (int i = 0; i < denoise_passes; i++) {
            denoise << <blocks, threads >> > (pbo_buffer, normals, positions, c_phi, n_phi, p_phi,
                offset, kernel, stepwidth, temp_buf, width, height);
            //checkCudaErrors(cudaMemcpy(pbo_buffer, temp_buf, width * height * sizeof(color), cudaMemcpyDeviceToDevice));
            
            c_phi *= 0.5f;
            stepwidth *= 2;

            denoise << <blocks, threads >> > (temp_buf, normals, positions, c_phi, n_phi, p_phi,
                offset, kernel, stepwidth, pbo_buffer, width, height);

            c_phi *= 0.5f;
            stepwidth *= 2;
        }
        
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

    glm::ivec2* offset;
    float* kernel;
    color* temp_buf;
    int denoise_passes;
};

