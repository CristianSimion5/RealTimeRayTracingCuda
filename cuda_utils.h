#pragma once

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

#include "pdf.h"

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

__device__ glm::vec3 get_color(const ray& r, const color& background, hittable_list** world, hittable_list **lights, int depth,
    curandState* local_rand_state) {
    hit_record rec;
    scatter_record srec;
    srec.is_specular = false;
    ray cur_ray = r;
    color cur_albedo(1.0f);
    color cur_emitted(0.0f);
    color emitted;
    float pdf_val;

    ray scattered;
    ray shadow_ray;

    for (int i = 0; i < depth; i++) {
        if ((*world)->hit(cur_ray, 0.001f, infinity, rec, local_rand_state)) {

            if (isnan(rec.p.x) || isnan(rec.p.y) || isnan(rec.p.z)) {
                printf("%f %f %f || %f %f %f || %f\n", cur_ray.origin().x, cur_ray.origin().y, cur_ray.origin().z,
                    cur_ray.direction().x, cur_ray.direction().y, cur_ray.direction().z, rec.t);
            }
            if (i == 0 || srec.is_specular) {
                emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
                cur_emitted += emitted * cur_albedo;
            }

            if (rec.mat_ptr->scatter(cur_ray, rec, srec, local_rand_state)) {
                if (srec.is_specular) {
                    cur_albedo *= srec.attenuation;
                    cur_ray = srec.specular_ray;
                    continue;
                }

                // Sample direct light
                hittable_pdf light_pdf((*lights), rec.p);
                auto dif = light_pdf.generate(local_rand_state, emitted);
                auto shadow_ray_dir = glm::normalize(dif);
                shadow_ray = ray(rec.p, shadow_ray_dir, cur_ray.time());
                pdf_val = light_pdf.value(shadow_ray.direction());

                // Visibility Tests
                float visibility = 1.0f;
                if (glm::dot(shadow_ray.direction(), rec.normal) < 0.0f) // Testing if light position is facing away from the hit surface
                    visibility = 0.0f;

                if ((*world)->any_hit(shadow_ray, 0.001f, glm::length(dif) - 0.01f, local_rand_state)) // Test for any intersection between point hit and light source
                    visibility = 0.0f;

                float direct_light = 0.0f;
                if (glm::abs(pdf_val) > 1e-4f)
                    direct_light = visibility * rec.mat_ptr->scattering_pdf(cur_ray, rec, shadow_ray) / pdf_val;
                cur_emitted += glm::min(srec.attenuation * direct_light * cur_albedo * emitted, 1.0f);
                
                // Sample indirect light
                scattered = ray(rec.p, srec.pdf_obj.generate(local_rand_state, emitted), cur_ray.time());
                pdf_val = srec.pdf_obj.value(scattered.direction());
                float indirect_light = 0.0f;
                if (glm::abs(pdf_val) > 1e-5f)
                    indirect_light = rec.mat_ptr->scattering_pdf(cur_ray, rec, scattered) / pdf_val;
                 
                // Handle perpendicular grazing angles
                if (glm::abs(indirect_light - 1.0f) > 1e-5) {
                    //printf("%f %f\n", indirect_light, pdf_val);
                    return cur_emitted;
                }
               
                // Recursive call should be: 
                // return srec.attenuation * (direct_light * emitted_light + indirect_light * get_color(scattered, ... ))
                cur_albedo *=  srec.attenuation * indirect_light;

                // Update current ray for next iteration
                cur_ray = scattered;
            }
            else {
                return cur_emitted;
            }
        }
        else {
            return cur_emitted + cur_albedo * background;
        }
    }
    return cur_emitted; // Max depth reached
}

__device__ glm::vec3 get_first_bounce_data(const ray& r, const color& background, hittable_list** world, hittable_list** lights, int depth,
    glm::vec3& normal, point3& position, curandState* local_rand_state) {
    hit_record rec;
    scatter_record srec;
    ray cur_ray = r;
    color cur_albedo(1.0f);
    color cur_emitted(0.0f);
    color emitted;
    float pdf_val;

    ray scattered;
    ray shadow_ray;

    if ((*world)->hit(cur_ray, 0.001f, infinity, rec, local_rand_state)) {
        // First bounce special case, also save normal and position at hit point
        normal = rec.normal;
        position = rec.p;

        // If we hit an emissive material (aka a light) with the first bounce we explicitly add the emission
        emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
        cur_emitted += emitted * cur_albedo;

        if (rec.mat_ptr->scatter(cur_ray, rec, srec, local_rand_state)) {
            if (srec.is_specular) {
                cur_albedo *= srec.attenuation;
                cur_ray = srec.specular_ray;
            }
            else {
                // Sample direct light
                hittable_pdf light_pdf((*lights), rec.p);
                auto dif = light_pdf.generate(local_rand_state, emitted);
                auto shadow_ray_dir = glm::normalize(dif);
                shadow_ray = ray(rec.p, shadow_ray_dir, cur_ray.time());
                pdf_val = light_pdf.value(shadow_ray.direction());

                // Visibility Tests
                float visibility = 1.0f;
                if (glm::dot(shadow_ray.direction(), rec.normal) < 0.0f) // Testing if light position is facing away from the hit surface
                    visibility = 0.0f;

                if ((*world)->any_hit(shadow_ray, 0.001f, glm::length(dif) - 0.01f, local_rand_state)) // Test for any intersection between point hit and light source
                    visibility = 0.0f;

                float direct_light = 0.0f;
                if (glm::abs(pdf_val) > 1e-4f)
                    direct_light = visibility * rec.mat_ptr->scattering_pdf(cur_ray, rec, shadow_ray) / pdf_val;
                
                cur_emitted += glm::min(srec.attenuation * direct_light * cur_albedo * emitted, 1.0f);
                
                // Sample indirect light
                scattered = ray(rec.p, srec.pdf_obj.generate(local_rand_state, emitted), cur_ray.time());
                pdf_val = srec.pdf_obj.value(scattered.direction());
                float indirect_light = 0.0f;
                if (glm::abs(pdf_val) > 1e-5f)
                    indirect_light = rec.mat_ptr->scattering_pdf(cur_ray, rec, scattered) / pdf_val;

                // Handle perpendicular grazing angles
                if (glm::abs(indirect_light - 1.0f) > 1e-5f) {
                    //printf("first bounce: %f %f %f || %f %f %f || %f %f %f\n", rec.p.x, rec.p.y, rec.p.z, 
                        //rec.normal.x, rec.normal.y, rec.normal.z, 
                        //scattered.direction().x, scattered.direction().y, scattered.direction().z);
                    return cur_emitted;
                }

                // Recursive call should be: 
                // return srec.attenuation * (direct_light * emitted_light + indirect_light * get_color(scattered, ... ))
                cur_albedo *= srec.attenuation * indirect_light;
                cur_ray = scattered;
            }
        } else {
            return cur_emitted;
        }
    } else {
        // No bounce special case, normals and position is undefined
        normal = glm::vec3(0.5f);// glm::vec3(0, 0, 1);//glm::vec3(0.0f); // can also use normalized value like (0, 0, 1) if we get an error
        position = cur_ray.at(10000.0f);//point3(0.0f);  // If it doesn't work also try ray.at(large_value);

        return cur_emitted + cur_albedo * background;
    }

    for (int i = 1; i < depth; i++) {
        if ((*world)->hit(cur_ray, 0.001f, infinity, rec, local_rand_state)) {

            // Specular materials do not sample lights directly
            if (srec.is_specular) {
                emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
                cur_emitted += emitted * cur_albedo;
            }

            if (rec.mat_ptr->scatter(cur_ray, rec, srec, local_rand_state)) {
                if (srec.is_specular) {
                    cur_albedo *= srec.attenuation;
                    cur_ray = srec.specular_ray;
                    continue;
                }

                // Sample direct light
                hittable_pdf light_pdf((*lights), rec.p);
                auto dif = light_pdf.generate(local_rand_state, emitted);
                auto shadow_ray_dir = glm::normalize(dif);
                shadow_ray = ray(rec.p, shadow_ray_dir, cur_ray.time());
                pdf_val = light_pdf.value(shadow_ray.direction());

                // Visibility Tests
                float visibility = 1.0f;
                if (glm::dot(shadow_ray.direction(), rec.normal) < 0.0f) // Testing if light position is facing away from the hit surface
                    visibility = 0.0f;

                if ((*world)->any_hit(shadow_ray, 0.001f, glm::length(dif) - 0.01f, local_rand_state)) // Test for any intersection between point hit and light source
                    visibility = 0.0f;

                float direct_light = 0.0f;
                if (glm::abs(pdf_val) > 1e-4f)
                    direct_light = visibility * rec.mat_ptr->scattering_pdf(cur_ray, rec, shadow_ray) / pdf_val;
                cur_emitted += glm::min(srec.attenuation * direct_light * cur_albedo * emitted, 1.0f);
                
                // Sample indirect light
                scattered = ray(rec.p, srec.pdf_obj.generate(local_rand_state, emitted), cur_ray.time());
                pdf_val = srec.pdf_obj.value(scattered.direction());
                float indirect_light = 0.0f;
                if (glm::abs(pdf_val) > 1e-5f)
                    indirect_light = rec.mat_ptr->scattering_pdf(cur_ray, rec, scattered) / pdf_val;
                 
                // Handle perpendicular grazing angles
                if (glm::abs(indirect_light - 1.0f) > 1e-5) {
                    //printf("%f %f\n", indirect_light, pdf_val);
                    return cur_emitted;
                }
               
                // Recursive call should be: 
                // return srec.attenuation * (direct_light * emitted_light + indirect_light * get_color(scattered, ... ))
                cur_albedo *=  srec.attenuation * indirect_light;

                // Update current ray for next iteration
                cur_ray = scattered;
            }
            else {
                return cur_emitted;
            }
        }
        else {
            return cur_emitted + cur_albedo * background;
        }
    }
    return cur_emitted; // Max depth reached
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void create_world(hittable** d_list, int objects_size, hittable_list** d_world,
    hittable_list** d_lights, int num_lights,
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
        hittable** lights_list = new hittable*[num_lights];

        auto red = new lambertian(color(0.65f, 0.05f, 0.05f));
        auto white = new lambertian(color(0.73f, 0.73f, 0.73f));
        auto green = new lambertian(color(0.12f, 0.45f, 0.15f));
        auto light = new diffuse_light(color(15.f, 15.f, 15.f));
        //auto light = new diffuse_light(color(7.f, 7.f, 7.f));

        d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);
        d_list[1] = new yz_rect(0, 555, 0, 555, 0, red);
        d_list[2] = new xz_rect(213, 343, 227, 332, 554, light);
        //d_list[2] = new xz_rect(113, 443, 127, 432, 554, light);

        lights_list[0] = d_list[2];

        d_list[3] = new xz_rect(0, 555, 0, 555, 0, white);
        d_list[4] = new xz_rect(0, 555, 0, 555, 555, white);
        d_list[5] = new xy_rect(0, 555, 0, 555, 555, white);

        //auto aluminum = new metal(color(0.8f, 0.85f, 0.88f), 0.0f);
        hittable* box1 = new box(point3(0, 0, 0), point3(165, 330, 165), white);
        box1 = new rotate_y(box1, 15.0f);
        box1 = new translate(box1, glm::vec3(265, 0, 295));
        //box1 = new constant_medium(box1, 0.01f, color(0.0f));
        d_list[6] = box1;

        //hittable* box2 = new box(point3(0, 0, 0), point3(165, 165, 165), white);
        //box2 = new rotate_y(box2, -18.0f);
        //box2 = new translate(box2, glm::vec3(130, 0, 65));
         //box2 = new constant_medium(box2, 0.01f, color(1.0f));
         //d_list[7] = box2;
        auto glass = new dielectric(1.5f);
        //auto purple_aluminum = new metal(color(0.8f, 0.15f, 0.88f), 0.1f);
        d_list[7] = new sphere(point3(190, 90, 190), 90.0f, glass);
        // d_list[7] = new sphere(point3(200, 100, 200), 100.0f, purple_aluminum);
        // lights_list[1] = d_list[7];



        /*hittable_list boxes1;
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
        */
        *d_world = new hittable_list(d_list, objects_size);
        *d_lights = new hittable_list(lights_list, num_lights);

        //point3 origin(13, 2, 3);
        //point3 origin(26, 3, 6);
        point3 origin(278, 278, -800);
        //point3 origin(478, 278, -600);
        //point3 lookat(0, 0/*2*/, 0);
        point3 lookat(278, 278, 0);
        float aperture = 0.0f;
        float dist_to_focus = 20.0f; //glm::distance(origin, lookat);
        float vfov = 40.0f; //20.0f;
        //*d_camera = new camera(origin, lookat, glm::vec3(0, 1, 0), vfov,
                    //aspect_ratio, aperture, dist_to_focus, 0.0f, 1.0f);
        (*d_camera)->origin = origin;
        (*d_camera)->lookat = lookat;
        (*d_camera)->vup = glm::vec3(0, 1, 0);
        (*d_camera)->vfov = vfov;
        (*d_camera)->aspect = aspect_ratio;
        (*d_camera)->aperture = aperture;
        (*d_camera)->focus_dist = dist_to_focus;
        (*d_camera)->time0 = 0.0f;
        (*d_camera)->time1 = 1.0f;
        (*d_camera)->update_camera();
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(color* fb, color* pbo_buf, glm::vec3* normals, point3* positions, 
    int max_x, int max_y, int spp, int max_depth,
    camera** cam, color bgcolor, hittable_list** world, hittable_list** lights,
    curandState* rand_state, int frame_count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState& local_rand_state = rand_state[pixel_index];
    color col(0.0f);

    // First sample
    float u = float(i + curand_uniform(&local_rand_state)) / max_x;
    float v = float(j + curand_uniform(&local_rand_state)) / max_y;
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    //col += get_first_bounce_data(r, bgcolor, world, lights, max_depth, 
    //    normals[pixel_index], positions[pixel_index], &local_rand_state);

    for (int s = 0; s < spp; s++) {
        u = float(i + curand_uniform(&local_rand_state)) / max_x;
        v = float(j + curand_uniform(&local_rand_state)) / max_y;
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += get_color(r, bgcolor, world, lights, max_depth, &local_rand_state);
    }


    /*
    // Check for NaNs
    if (glm::any(glm::isnan(col))) {
        printf("nan detected\n");
    }
    if (col.r != col.r) col.r = 0.0f;
    if (col.g != col.g) col.g = 0.0f;
    if (col.b != col.b) col.b = 0.0f;
    */

    //col = glm::clamp(col, 0.0f, 1.0f);

    if (frame_count > 1) {
        fb[pixel_index] += col;
    }
    else {
        fb[pixel_index] = col;
    }

    pbo_buf[pixel_index] = glm::sqrt(fb[pixel_index] / (float(spp) * frame_count));
    //pbo_buf[pixel_index] = normals[pixel_index] / 2.f + 0.5f;
}

__global__ void denoise(color* src, glm::vec3* normals, point3* positions,
    float c_phi, float n_phi, float p_phi,
    glm::ivec2* offset, float* kernel, int stepwidth,
    color* temp_buf, int max_x, int max_y) {
 
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    // i - coloana (directia pe x), j - linia (directia pe y) 
    int pixel_index = j * max_x + i;
    color col = src[pixel_index], new_col;
    glm::vec3 nrm = normals[pixel_index], new_nrm;
    glm::vec3 pos = positions[pixel_index], new_pos;

    glm::vec3 s(0.0f);
    float k = 0.0f;
    float dst, c_w, n_w, p_w, w;
    
    for (int ind = 0; ind < 25; ind++) {
        auto q = glm::ivec2(i, j) + offset[ind] * stepwidth;
        if (q.x < 0 || q.x >= max_x || q.y < 0 || q.y >= max_y)
            continue;

        int new_index = q.y * max_x + q.x;
        new_col = src[new_index];
        new_nrm = normals[new_index];
        new_pos = positions[new_index];

        dst = glm::distance(col, new_col);
        c_w = glm::min(glm::exp(-dst / c_phi), 1.0f);

        dst = glm::distance(nrm, new_nrm);
        n_w = glm::min(glm::exp(-dst / n_phi), 1.0f);

        dst = glm::distance(pos, new_pos);
        p_w = glm::min(glm::exp(-dst / p_phi), 1.0f);

        w = c_w * n_w * p_w;
        s += kernel[ind] * w * new_col;
        k += kernel[ind] * w;
    }

    temp_buf[pixel_index] = s / k;
}

__global__ void free_world(hittable** d_list, int objects_size, hittable_list** d_world, hittable_list** d_lights) {
    for (int i = 0; i < objects_size; i++) {
        delete d_list[i];
    }
    delete *d_world;
    delete[] (*d_lights)->objects;
    delete *d_lights;
}

