#pragma once

#include <GL/glew.h>

#include <driver_types.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include <cstdio>

#include "shader.h"
#include "cuda_utils.h"
#include "scene_settings.h"

void GLAPIENTRY
MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
        type, severity, message);
}

class gl_wrapper {
public:
    gl_wrapper(scene_settings& sc, const char* vshaderfn, const char* fshaderfn, bool debug = true) 
        : basic_shader(vshaderfn, fshaderfn), scene(sc) {
        if (debug) {
            // During init, enable debug output
            glEnable(GL_DEBUG_OUTPUT);
            glDebugMessageCallback(MessageCallback, 0);
        }

        fb_size = (size_t) scene_settings::MAX_PIXELS * sizeof(color);
        checkCudaErrors(cudaMallocManaged((void**) &fb, fb_size));

        GLfloat vertices[] = {
           -1.f, -1.f, 0.f, 0.f, 0.f,      // format: x, y, z, u, v
            1.f, -1.f, 0.f, 1.f, 0.f,
            1.f,  1.f, 0.f, 1.f, 1.f,
           -1.f,  1.f, 0.f, 0.f, 1.f
        };

        GLuint indices[] = {
            0, 1, 2,
            2, 3, 0
        };

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // Vertex positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), 0);

        glEnableVertexAttribArray(1); // Texture coordinates
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

        glGenBuffers(1, &ibo);  // Index Buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
        glDisable(GL_DEPTH_TEST);

        glUseProgram(basic_shader.id);
        glUniform1i(glGetUniformLocation(basic_shader.id, "rt_image"), 0);
        glUseProgram(0);
        glBindVertexArray(0);   // Unbind VAO
    
        glGenTextures(1, &tex_obj);
        glBindTexture(GL_TEXTURE_2D, tex_obj);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scene.width, scene.height, 0, GL_RGB, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, fb_size, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
     
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_resource_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    }

    ~gl_wrapper() {
        // Raises CUDA error, probably cudaDeviceReset already unregisters the resource (or some other destructive call)
        //checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource_pbo));
        checkCudaErrors(cudaFree(fb));
    
        glDeleteBuffers(1, &pbo);
        glDeleteBuffers(1, &ibo);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);

        glDeleteTextures(1, &tex_obj);

        glDeleteProgram(basic_shader.id);
    }

    void render_world() {
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_pbo, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void**)&pbo_buffer, &pbo_buffer_size, cuda_resource_pbo));
        
        if (fb_size != pbo_buffer_size) {
            std::cout << "Size mismatch: " << fb_size << ' ' << pbo_buffer_size << '\n';
        }

        scene.generate_frame(fb, pbo_buffer);
    
        // Wait for the frame to render, probably unnecessary since UnmapResources also synchronizes
        // checkCudaErrors(cudaGetLastError());
        // checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_pbo, 0));

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(basic_shader.id);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_obj);

        // Calling glTexSubImage2D when you have a PBO bound treats the PBO 
        // as the data buffer and starts a pixel unpack operation
        // Note: rendered texture may not be 1:1 with the generated image
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, scene.width, scene.height, GL_RGB, GL_FLOAT, NULL);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, NULL);
        glBindVertexArray(0);

        glUseProgram(0);
    }

    void update_viewport(int new_width, int new_height) {
        glBindTexture(GL_TEXTURE_2D, tex_obj);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, new_width, new_height, 0, GL_RGB, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        (**scene.d_camera).aspect = 1.0f * new_width / new_height;
        (**scene.d_camera).update_camera();

        glViewport(0, 0, new_width, new_height);
        scene.width = new_width;
        scene.height = new_height;

        scene.continuous_frame_count = 1;
    }
    
    scene_settings& scene;

private:
    GLuint vao, vbo, ibo;
    GLuint tex_obj;
    GLuint pbo;
    shader basic_shader;

    cudaGraphicsResource* cuda_resource_pbo;
    color* pbo_buffer;
    size_t pbo_buffer_size;

    color* fb;
    size_t fb_size;
};

