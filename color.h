#pragma once

#include <iostream>

#include "vector3.h"
#include "stb/stb_image_write_implementation.h"

void write_color(std::ostream& out, color pixel_color, int samples_per_pixel) {
    // Divide to account for multisampling and gamma-correct for gamma 2
    auto scale = 1.0f / samples_per_pixel;
    auto rgb = sqrt(pixel_color * scale);
    rgb = clamp(rgb, 0.0f, 0.999f);

    out << static_cast<int>(256 * rgb.r) << ' '
        << static_cast<int>(256 * rgb.g) << ' '
        << static_cast<int>(256 * rgb.b) << '\n';
}

void write_to_file(const char* filename, int width, int height, color* image_data, int samples_per_pixel, bool flip_y = true, bool gamma_correct = true) {
    auto scale = 1.0f / samples_per_pixel;
    unsigned char* new_data = new unsigned char[width * height * 3];
    color rgb;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            rgb = image_data[i * width + j];
            for (int c = 0; c < 3; c++) {
                if (gamma_correct)
                    rgb[c] = sqrt(rgb[c] * scale);
                else
                    rgb[c] = rgb[c] * scale;
                rgb[c] = glm::clamp(rgb[c], 0.0f, 0.999f);
                new_data[i * width * 3 + j * 3 + c] = static_cast<unsigned char>(256 * rgb[c]);
            }
        }
    }

    if (flip_y)
        stbi_flip_vertically_on_write(1);
    else
        stbi_flip_vertically_on_write(0);

    stbi_write_bmp(filename, width, height, 3, new_data);
    delete[] new_data;
}

void write_to_file_hdr(const char* filename, int width, int height, float*** image_data, int samples_per_pixel, bool flip_y = true) {
    auto scale = 1.0f / samples_per_pixel;
    float* new_data = new float[width * height * 3];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            auto rgb = image_data[i][j];
            for (int c = 0; c < 3; c++) {
                rgb[c] = rgb[c] * scale;
                new_data[i * width * 3 + j * 3 + c] = rgb[c];
            }
        }
    }

    if (flip_y)
        stbi_flip_vertically_on_write(1);
    else
        stbi_flip_vertically_on_write(0);

    stbi_write_hdr(filename, width, height, 3, new_data);
    delete[] new_data;
}

void write_binary(const char* filename, int width, int height, float*** image_data, int samples_per_pixel) {
    auto scale = 1.0f / samples_per_pixel;
    FILE* f = fopen(filename, "wb");

    fwrite(&width, sizeof(width), 1, f);
    fwrite(&height, sizeof(height), 1, f);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            auto rgb = image_data[i][j];
            for (int c = 0; c < 3; c++) {
                rgb[c] = rgb[c] * scale;
            }
            fwrite(rgb, sizeof(float), 3, f);
        }
    }

    fclose(f);
}