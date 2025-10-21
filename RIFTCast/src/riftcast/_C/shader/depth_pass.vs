#version 330 core

#define NORMAL_OFFSET 0.001

layout (location = 0) in vec3 aPosition;
layout (location = 2) in vec3 aNormal;

out vec4 frag_pos;

uniform mat4 V, P;

void main()
{
    frag_pos = P * V * vec4(aPosition + aNormal * NORMAL_OFFSET, 1);

    gl_Position = frag_pos;

}