#version 330 core

in vec2 uv;
uniform sampler2D rt_image;

out vec4 frag_color;

void main() {
    frag_color = texture(rt_image, uv);//vec4(uv, 0.0f, 1.0f);
}