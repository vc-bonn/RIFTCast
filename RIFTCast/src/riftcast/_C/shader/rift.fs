#version 330 core

#include "num_cameras.glsl"

layout (location = 0) out vec4 fragColor;
layout (location = 1) out int outEntityID;
layout (location = 2) out int visibility;
layout (location = 3) out vec4 out_depth;
layout (location = 4) out vec4 fragNormal;

in VS_OUT
{
    vec3 frag_pos;
    vec3 frag_projected[NUM_CAMERAS];
    vec3 frag_normal;
    vec3 frag_projected_mapping;
} fs_in;

uniform sampler3D texture_tensor;
uniform vec3 cam_position_mapping;
uniform int entityID;
uniform sampler2D depth_maps[NUM_CAMERAS];
uniform vec3 cam_positions[NUM_CAMERAS];
uniform mat4 cam_view_projections[NUM_CAMERAS];

vec4 cubic(float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec3 read_interpolated(vec3 uv)
{
    
    vec2 texSize = textureSize(texture_tensor, 0).rg;
    vec2 invTexSize = 1.0 / texSize * 0.5;

    vec4 c00 = texture(texture_tensor, vec3(uv.xy + invTexSize * vec2(-1,-1), uv.z));
    vec4 c10 = texture(texture_tensor, vec3(uv.xy + invTexSize * vec2(1,-1), uv.z));
    vec4 c01 = texture(texture_tensor, vec3(uv.xy + invTexSize * vec2(-1,1), uv.z));
    vec4 c11 = texture(texture_tensor, vec3(uv.xy + invTexSize * vec2(1,1), uv.z));

    vec2 fxy = fract(uv.xy * texSize - 0.5);

    vec4 c0 = mix(c00, c10, fxy.x);
    vec4 c1 = mix(c01, c11, fxy.x);

    return mix(c0, c1, fxy.y).rgb;
}

vec3 read_color(vec3 frag_uv)
{
    vec3 uv = frag_uv.xyz;
    uv.z += 0.001;
    return read_interpolated(uv);
}

vec3 project_point(mat4 VP, vec3 point)
{
    vec4 image_space = VP * vec4(point, 1.0);
    vec3 projected = (image_space.xyz / image_space.w + 1.0) * 0.5;
    return projected;
}

void main()
{
    vec3 pixel_color = vec3(0);
    float pixel_weight = 0;

    vec3 view_dir_main = normalize(fs_in.frag_pos - cam_position_mapping);
    vec3 world_vertex = fs_in.frag_pos;

    vec3 colors[NUM_CAMERAS];
    float weights[NUM_CAMERAS];
    int num_visible = 0;
    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        weights[i] = 0;
        colors[i] = vec3(0);
        vec3 projected = project_point(cam_view_projections[i], world_vertex);
        vec3 view_dir =  normalize(cam_positions[i] - world_vertex);

        if(projected.x < 0. || projected.x > 1. || projected.y < 0. || projected.y > 1.) continue;

        float current_depth = projected.z;
        float closest_depth = texture(depth_maps[i], projected.xy).r;

        // if(/*current_depth - 6e-5 <= closest_depth &&*/ dot(fs_in.frag_normal, view_dir) > -1)
        if(current_depth - 6e-5 <= closest_depth && dot(fs_in.frag_normal, view_dir) > -1)
        {
            ++num_visible;

            // float weight = max(0., dot(fs_in.frag_normal, view_dir)) * max(0.0, -dot(view_dir, view_dir_main));
            float weight = pow(max(0., dot(fs_in.frag_normal, view_dir)),2.0) * pow(max(0.0, -dot(view_dir, view_dir_main)),16.0);
            colors[i] = read_color(vec3(projected.xy, float(i)/float(NUM_CAMERAS)));
            weights[i] = weight;
        }
    }

    for(int i = 0; i < NUM_CAMERAS; ++i)
    {
        pixel_weight += weights[i];
        pixel_color += weights[i] * colors[i];
    }
    pixel_color = pixel_color / pixel_weight;

    if(pixel_weight < 1e-5)
    {
        num_visible = 0;
        pixel_color = vec3(0);
    }

    fragColor = vec4(pixel_color, 1.0);
    outEntityID = entityID;
    visibility = num_visible;
    out_depth = vec4(vec3(gl_FragCoord.z), 1);
    fragNormal = vec4(fs_in.frag_normal, 1);
}