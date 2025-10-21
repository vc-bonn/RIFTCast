#version 330 core

layout (location = 0) out vec4 fragColor;

in vec4 frag_pos;

void main()
{
    float depth = (frag_pos.z / frag_pos.w + 1.0) * 0.5;
    fragColor = vec4(depth, depth, depth, 1.0);
}