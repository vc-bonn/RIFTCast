#version 330 core

layout (location = 0) out int primitiveID;

void main()
{
    primitiveID = gl_PrimitiveID;
}