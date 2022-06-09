#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "utility.h"

#include "camera.h"
#include "sphere.h"
#include "hittable_list.h"
#include "color.h"
#include "material.h"

#include "moving_sphere.h"
#include "bvh.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"

#include <iostream>
#include <chrono>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << ": " << 
            cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Need to call before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const point3& center, float radius, const ray& r) {
    glm::vec3 oc = r.origin() - center;
    float a = glm::dot(r.direction(), r.direction());
    float b = 2.0f * glm::dot(oc, r.direction());
    float c = glm::dot(oc, oc) - radius * radius;
    float det = b * b - 4.0f * a * c;
    return det > 0.0f;
}

__device__ glm::vec3 get_color(const ray& r, const color& background, hittable_list **world, int depth, 
    curandState* local_rand_state) {
    hit_record rec;
    ray cur_ray = r;
    color cur_attenuation(1.0f);
    color cur_emitted(0.0f);

    for (int i = 0; i < depth; i++) {
        if ((*world)->hit(cur_ray, 0.001f, infinity, rec, local_rand_state)) {
            ray scattered;
            color attenuation;
            color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            cur_emitted += emitted * cur_attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_ray = scattered;
                cur_attenuation *= attenuation;
            }
            else {
                return cur_emitted;
            }
        } else {
            return cur_emitted + cur_attenuation * background;
        }
    }
    return cur_emitted; // Max depth reached
}

__global__ void create_world(hittable** d_list, int objects_size, hittable_list** d_world,
    camera** d_camera, unsigned char** textures, image_params* tex_info, float aspect_ratio, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        /*
        auto ground_material = new checker_texture(color(0.2f, 0.3f, 0.1f), color(0.9f));//new lambertian(color(0.5f));
        d_list[0] = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(ground_material));

        int ob = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = random_float(rand_state);
                point3 center(a + 0.9f * random_float(rand_state), 0.2f, b + 0.9f * random_float(rand_state));

//                if (glm::distance(center, point3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                material* sphere_material;

                if (choose_mat < 0.8f) {
                    auto albedo = random(rand_state) * random(rand_state);
                    auto center2 = center + glm::vec3(0, random_float(0, 0.5f, rand_state), 0);
                    sphere_material = new lambertian(albedo);
                    d_list[ob] = new moving_sphere(center, center2, 0.0f, 1.0f, 0.2f, sphere_material);
                } else if (choose_mat < 0.95f) {
                    auto albedo = random(0.5f, 1.0f, rand_state);
                    auto fuzz = random_float(0.0f, 0.5f, rand_state);
                    sphere_material = new metal(albedo, fuzz);
                    d_list[ob] = new sphere(center, 0.2f, sphere_material);
                } else {
                    sphere_material = new dielectric(1.5f);
                    d_list[ob] = new sphere(center, 0.2f, sphere_material);
                }
                ob++;
//                }
            }
        }

        auto material1 = new dielectric(1.5f);//metal(color(0.8f, 0.8f, 0.8f), 0.3f);
        d_list[ob++] = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1);

        auto material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
        d_list[ob++] = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2);

        auto material3 = new metal(color(0.7f, 0.6f, 0.5f), 0.0f);
        d_list[ob++] = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, material3);
        */

        // textures
        //auto earth_surface = new lambertian(&textures[0]);
//        auto globe = new sphere(point3(0), 2, new lambertian(&textures[0]));

        //d_list[0] = new sphere(point3(0), 2, new lambertian(new image_texture(textures[1], tex_info[1])));//globe;
//        d_list[0] = new sphere(point3(0, 0, -1), 0.5f, new lambertian(color(0.4f, 0.2f, 0.1f)));
        //d_list[1] = new sphere(point3(0, -100.5f, -1), 100.0f, nullptr);
        
        /* perlin + diffuse_light
        auto pertext = new noise_texture(4.f, rand_state);
        d_list[0] = new sphere(point3(0, -1000.0f, 0), 1000.0f, new lambertian(pertext));
        d_list[1] = new sphere(point3(0, 2.0f, 0), 2.0f, new lambertian(pertext));
        
        auto difflight = new diffuse_light(color(4));
        d_list[2] = new xy_rect(3, 5, 1, 3, -2, difflight);
        d_list[3] = new sphere(point3(0, 7.0f, 0), 2.0f, new diffuse_light(color(4)));
        */

        /*auto red = new lambertian(color(0.65f, 0.05f, 0.05f));
        auto white = new lambertian(color(0.73f, 0.73f, 0.73f));
        auto green = new lambertian(color(0.12f, 0.45f, 0.15f));
        //auto light = new diffuse_light(color(15.f, 15.f, 15.f));
        auto light = new diffuse_light(color(7.f, 7.f, 7.f));

        d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);
        d_list[1] = new yz_rect(0, 555, 0, 555, 0, red);
        //d_list[2] = new xz_rect(213, 343, 227, 332, 554, light);
        d_list[2] = new xz_rect(113, 443, 127, 432, 554, light);
        d_list[3] = new xz_rect(0, 555, 0, 555, 0, white);
        d_list[4] = new xz_rect(0, 555, 0, 555, 555, white);
        d_list[5] = new xy_rect(0, 555, 0, 555, 555, white);

        hittable* box1 = new box(point3(0, 0, 0), point3(165, 330, 165), white);
        box1 = new rotate_y(box1, 15.0f);
        box1 = new translate(box1, glm::vec3(265, 0, 295));
        box1 = new constant_medium(box1, 0.01f, color(0.0f), rand_state);
        d_list[6] = box1;

        hittable* box2 = new box(point3(0, 0, 0), point3(165, 165, 165), white);
        box2 = new rotate_y(box2, -18.0f);
        box2 = new translate(box2, glm::vec3(130, 0, 65));
        box2 = new constant_medium(box2, 0.01f, color(1.0f), rand_state);
        d_list[7] = box2;
        */
        hittable_list boxes1;
        auto ground = new lambertian(color(0.48f, 0.83f, 0.53f));

        const int boxes_per_side = 20;
        boxes1.objects_size = boxes_per_side * boxes_per_side;
        boxes1.objects = new hittable*[boxes1.objects_size];

        for (int i = 0; i < boxes_per_side; i++) {
            for (int j = 0; j < boxes_per_side; j++) {
                auto w = 100.0f;
                auto x0 = -1000.0f + i * w;
                auto z0 = -1000.0f + j * w;
                auto y0 = 0.0f;
                auto x1 = x0 + w;
                auto y1 = random_float(1, 101, rand_state);
                auto z1 = z0 + w;

                boxes1.objects[i * boxes_per_side + j] =
                    new box(point3(x0, y0, z0), point3(x1, y1, z1), ground);
            }
        }
        //d_list[0] = new bvh_node(boxes1, 0.0f, 1.0f, rand_state);
        d_list[0] = new hittable_list(boxes1.objects, boxes1.objects_size);

        auto light = new diffuse_light(color(7.0f));
        d_list[1] = new xz_rect(123, 423, 147, 412, 554, light);

        auto center1 = point3(400.0f);
        auto center2 = center1 + glm::vec3(30.0f, 0, 0);
        auto moving_sphere_material = new lambertian(color(0.7f, 0.3f, 0.1f));
        d_list[2] = new moving_sphere(center1, center2, 0.0f, 1.0f, 50.0f, moving_sphere_material);

        d_list[3] = new sphere(point3(260, 150, 45), 50.0f, new dielectric(1.5f));
        d_list[4] = new sphere(point3(0, 150, 145), 50.0f, new metal(color(0.8f, 0.8f, 0.9f), 1.0f));

        auto boundary = new sphere(point3(360, 150, 145), 70.0f, new dielectric(1.5f));
        d_list[5] = boundary;
        d_list[6] = new constant_medium(boundary, 0.2f, color(0.2f, 0.4f, 0.9f));
        boundary = new sphere(point3(0.0f), 5000.0f, new dielectric(1.5f));
        d_list[7] = new constant_medium(boundary, 0.0001f, color(1.0f));

        auto emat = new lambertian(new image_texture(textures[0], tex_info[0]));
        d_list[8] = new sphere(point3(400.0f, 200.0f, 400.0f), 100.0f, emat);
        auto pertext = new noise_texture(0.1f, rand_state);
        d_list[9] = new sphere(point3(220, 280, 300), 80.0f, new lambertian(pertext));

        hittable_list boxes2;
        auto white = new lambertian(color(0.73f));
        boxes2.objects_size = 1000;
        boxes2.objects = new hittable*[boxes2.objects_size];
        for (int j = 0; j < boxes2.objects_size; j++) {
            boxes2.objects[j] = new sphere(random(0.0f, 165.0f, rand_state), 10.0f, white);
        }

        d_list[10] = new translate(
            new rotate_y(
                new hittable_list(boxes2.objects, boxes2.objects_size), 
                15.0f
            ),
            glm::vec3(-100, 270, 395)    
        );

        *d_world  = new hittable_list(d_list, objects_size);

        //point3 origin(13, 2, 3);
        //point3 origin(26, 3, 6);
        //point3 origin(278, 278, -800);
        point3 origin(478, 278, -600);
        //point3 lookat(0, 0/*2*/, 0);
        point3 lookat(278, 278, 0);
        float aperture = 0.0f;
        auto dist_to_focus = 20.0f; //glm::distance(origin, lookat);
        float vfov = 40.0f; //20.0f;
        *d_camera = new camera(origin, lookat, glm::vec3(0, 1, 0), vfov,
                    aspect_ratio, aperture, dist_to_focus, 0.0f, 1.0f);
    }
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(color* fb, int max_x, int max_y, int spp, int max_depth,
    camera** cam, color bgcolor, hittable_list** world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col(0.0f);

    for (int s = 0; s < spp; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / max_x;
        float v = float(j + curand_uniform(&local_rand_state)) / max_y;
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += get_color(r, bgcolor, world, max_depth, &local_rand_state);
    }
    fb[pixel_index] = col;// / float(spp);
}

__global__ void free_world(hittable** d_list, int objects_size, hittable_list** d_world, camera** d_camera) {
    for (int i = 0; i < objects_size; i++) {
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc, char* argv[]) {

    int nx = 800, ny = 800, spp = 50, max_depth = argc == 2 ? atoi(argv[1]) : 8;
    color bgcolor(0.0f);//(0.7f, 0.8f, 1.0f);
    float aspect_ratio = float(nx) / ny;
    int objects_size = 11;//4 + 22 * 22;
    int num_pixels = nx * ny;
    
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**) &d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state_world_gen;
    checkCudaErrors(cudaMalloc((void**) &d_rand_state_world_gen, sizeof(curandState)));
    rand_init<<<1, 1>>>(d_rand_state_world_gen);

    size_t fb_size = (size_t) num_pixels * sizeof(color); 
    color* fb;
    checkCudaErrors(cudaMallocManaged((void **) &fb, fb_size));

    int num_textures = 2;
    unsigned char** image_textures;
    checkCudaErrors(cudaMallocManaged((void**) &image_textures, num_textures * sizeof(unsigned char*)));
    image_params* image_info;
    checkCudaErrors(cudaMallocManaged(&image_info, num_textures * sizeof(image_params)));

    load_texture_data("../../textures/earthmap.jpg", image_textures, image_info, 0);
    load_texture_data("../../textures/happyface.jpg", image_textures, image_info, 1);

    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**) &d_list, objects_size * sizeof(hittable*)));
    hittable_list** d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable_list*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**) &d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (d_list, objects_size, d_world, d_camera, image_textures, image_info,
        aspect_ratio, d_rand_state_world_gen);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    int tx = 8, ty = 8;

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<< <blocks, threads >> >(fb, nx, ny, spp, max_depth,
        d_camera, bgcolor, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Done! Writing to image...";
    /*std::cout << "P3\n" << nx << ' ' << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = (size_t) j * nx + (size_t) i;

            float r = fb[pixel_index].r;
            float g = fb[pixel_index].g;
            float b = fb[pixel_index].b;
            
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);

            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    */
    write_to_file("final4.bmp", nx, ny, fb, spp);

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, objects_size, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(image_info));
    checkCudaErrors(cudaFree(image_textures));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
    return 0;

    
    //// Image
    //auto aspect_ratio = 16.0 / 9.0;
    //int image_width = 400;
    //int samples_per_pixel = 100;
    //const int max_depth = 50;

    ////std::vector<std::vector<glm::vec3>> image(image_height, std::vector<glm::vec3>(image_width));
    //float*** image = new float** [image_height];  
    //for (int i = 0; i < image_height; i++) {
    //    image[i] = new float* [image_width];
    //    for (int j = 0; j < image_width; j++) {
    //        image[i][j] = new float[3];
    //    }
    //}
}