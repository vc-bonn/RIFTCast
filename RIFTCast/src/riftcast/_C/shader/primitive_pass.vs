#version 330 core

layout (location = 0) in vec3 aPosition;

uniform mat4 V, P;

void main()
{
    vec3 frag_pos = vec3(vec4(aPosition, 1));

    gl_Position = P * V * vec4(frag_pos, 1);

}