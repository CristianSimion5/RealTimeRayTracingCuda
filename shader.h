#pragma once

#include <gl/glew.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

class shader {
public:
    shader(const char* vertPath, const char* fragPath) {
        std::string vCode;
        std::string fCode;
    
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;

        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try {
            vShaderFile.open(vertPath);
            fShaderFile.open(fragPath);
            std::stringstream vShaderStream, fShaderStream;

            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();

            vShaderFile.close();
            fShaderFile.close();
            
            vCode = vShaderStream.str();
            fCode = fShaderStream.str();
        }
        catch (std::ifstream::failure e) {
            std::cerr << "Error in shader: file was not read successfully'\n";
        }

        const char* vShaderCode = vCode.c_str();
        const char* fShaderCode = fCode.c_str();
        
        unsigned int vert, frag;
        int success;
        char infoLog[512];

        vert = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 1, &vShaderCode, NULL);
        glCompileShader(vert);
        glGetShaderiv(vert, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vert, 512, NULL, infoLog);
            std::cerr << "Error in vertex shader, compilation failed: " << infoLog << '\n';
        }

        frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &fShaderCode, NULL);
        glCompileShader(frag);
        glGetShaderiv(frag, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(frag, 512, NULL, infoLog);
            std::cerr << "Error in fragment shader, compilation failed: " << infoLog << '\n';
        }

        id = glCreateProgram();
        glAttachShader(id, vert);
        glAttachShader(id, frag);
        glLinkProgram(id);

        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(id, 512, NULL, infoLog);
            std::cerr << "Error in shader, linking failed: " << infoLog << '\n';
        }

        glDeleteShader(vert);
        glDeleteShader(frag);
    }

public:
    unsigned int id;
};

