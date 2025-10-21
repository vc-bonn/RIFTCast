#version 330 core

#include "num_cameras.glsl"

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aUV;
layout (location = 5) in float aDisplacement;

out VS_OUT
{
    vec3 frag_pos;
    vec3 frag_projected[NUM_CAMERAS];
    vec3 frag_normal;
    vec3 frag_projected_mapping;
} vs_out;

uniform mat4 V, P;
uniform mat4 cam_view_projection_mapping;
uniform mat4 cam_view_projections[NUM_CAMERAS];

vec3 project_point(mat4 VP, vec3 point)
{
    vec4 image_space = VP * vec4(point, 1.0);
    vec3 projected = (image_space.xyz / image_space.w + 1.0) * 0.5;
    return projected;
}

void main()
{
    // Gradient descent
    vec3 current_x = vec3(vec4(aPosition, 1));
    
    vs_out.frag_pos = current_x;

    gl_Position = P * V * vec4(vs_out.frag_pos, 1);

    vs_out.frag_normal = normalize((vec4(aNormal, 0.)).xyz);
    vs_out.frag_projected_mapping = project_point(cam_view_projection_mapping, current_x);

    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        vs_out.frag_projected[i] = project_point(cam_view_projections[i], vs_out.frag_pos);
    }
}