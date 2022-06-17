#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "utility.h"
#include "cuda_utils.h"
#include "scene_settings.h"

#include "gl_wrapper.h"
#include "shader.h"

#include <iostream>
#include <chrono>

gl_wrapper* gl_ptr;

void error_callback(int error, const char* description)
{
    std::cerr << "Error: " << description << '\n';
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    if (key == GLFW_KEY_EQUAL && action == GLFW_PRESS) {
        gl_ptr->scene.denoise_passes++;
        std::cout << "Increased denoising passes to " << gl_ptr->scene.denoise_passes << '\n';
    }

    if (key == GLFW_KEY_MINUS && action == GLFW_PRESS) {
        gl_ptr->scene.denoise_passes--;
        gl_ptr->scene.denoise_passes = glm::max(gl_ptr->scene.denoise_passes, 0);
        std::cout << "Decreased denoising passes to " << gl_ptr->scene.denoise_passes << '\n';
    }

    if (key == GLFW_KEY_LEFT && action == GLFW_REPEAT) {
        gl_ptr->scene.move_camera(glm::vec3(10, 0, 0));
    }
    if (key == GLFW_KEY_RIGHT && action == GLFW_REPEAT) {
        gl_ptr->scene.move_camera(glm::vec3(-10, 0, 0));
    }
    if (key == GLFW_KEY_UP && action == GLFW_REPEAT) {
        gl_ptr->scene.move_camera(glm::vec3(0, 10, 0));
    }
    if (key == GLFW_KEY_DOWN && action == GLFW_REPEAT) {
        gl_ptr->scene.move_camera(glm::vec3(0, -10, 0));
    }
}

void resize_callback(GLFWwindow*, int width, int height) {
    int w = glm::min(width, scene_settings::MAX_WIDTH);
    int h = glm::min(height, scene_settings::MAX_HEIGHT);

    gl_ptr->update_viewport(w, h);
}

int main(int argc, char* argv[]) {
    scene_settings scene(600, 500, 1, 2, color(0.0f), 8, 2, 1);

    if (!glfwInit()) {
        std::cerr << "Error: GLFW could not be initialized...";
        return 99;
    }

    glfwSetErrorCallback(error_callback);

    GLFWwindow* window = glfwCreateWindow(scene.width, scene.height, "Real Time Ray Tracing CUDA", NULL, NULL);
    if (!window) {
        std::cerr << "Error: Window creation failed...";
        glfwTerminate();
        return 99;
    }
    glfwMakeContextCurrent(window);

    // glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Error: GLEW failed to initialize..." << std::endl;
        return 99;
    }

    gl_wrapper opengl_wrap(scene,
        "vertex_shader.vert", "fragment_shader.frag");
    gl_ptr = &opengl_wrap;
    
    glfwSetFramebufferSizeCallback(window, resize_callback);
    glfwSetKeyCallback(window, key_callback);

    while (!glfwWindowShouldClose(window)) {
        opengl_wrap.render_world();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}