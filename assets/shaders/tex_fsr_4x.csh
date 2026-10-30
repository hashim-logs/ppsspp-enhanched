// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2024- PPSSPP Project.
// FSR 1.0 EASU (Edge-Adaptive Spatial Upscaling) compute shader for texture upscaling (4x)
// Adapted from AMD's FSR 1.0 reference implementation (MIT License)

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Output image (4x scaled)
layout(set = 0, binding = 0, rgba8) writeonly image2D dstImg;

// Input buffer (original texture data)
layout(std430, set = 0, binding = 1) readonly buffer Buf {
    uint data[];
} buf;

// Use the wrapper template's Params (width, height, enableAIUpscaling)
// width/height = source dimensions, dst is 4x scale
layout(push_constant) uniform Params {
    int width;
    int height;
    int enableAIUpscaling;  // 1 = FSR EASU, 0 = bilinear fallback
} params;

#define srcWidth params.width
#define srcHeight params.height
#define dstWidth (params.width * 4)
#define dstHeight (params.height * 4)

// FSR EASU constants
const float FSR_EPS = 0.0001;

// Read source pixel (clamped)
vec4 readSrc(ivec2 p) {
    p = clamp(p, ivec2(0, 0), ivec2(srcWidth - 1, srcHeight - 1));
    uint packed = buf.data[p.y * srcWidth + p.x];
    return unpackUnorm4x8(packed);
}

// Luminance coefficients (Rec.709)
const vec3 lumCoef = vec3(0.2126, 0.7152, 0.0722);

float Luminance(vec3 rgb) {
    return dot(rgb, lumCoef);
}

// FSR EASU core functions (adapted from fragment shader)
void FsrEasuCon(
    out vec4 con0, out vec4 con1, out vec4 con2, out vec4 con3,
    vec2 inputViewportInPixels,
    vec2 inputSizeInPixels,
    vec2 outputSizeInPixels)
{
    con0 = vec4(
        inputViewportInPixels.x / outputSizeInPixels.x,
        inputViewportInPixels.y / outputSizeInPixels.y,
        0.5 * inputViewportInPixels.x / outputSizeInPixels.x - 0.5,
        0.5 * inputViewportInPixels.y / outputSizeInPixels.y - 0.5);

    con1 = vec4(1.0, 1.0, 1.0, -1.0) / inputSizeInPixels.xyxy;
    con2 = vec4(-1.0,  2.0,  1.0,  2.0) / inputSizeInPixels.xyxy;
    con3 = vec4( 0.0,  4.0,  0.0,  0.0) / inputSizeInPixels.xyxy;
}

void FsrEasuTapF(
    inout vec3 aC, inout float aW,
    vec2 off,
    vec2 dir, vec2 len,
    float lob, float clp,
    vec3 c)
{
    vec2 v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x))) * len;
    float d2 = min(dot(v, v), clp);
    float wB = 0.4 * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB; wA *= wA;
    wB = 1.5625 * wB - 0.5625;
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

void FsrEasuSetF(
    inout vec2 dir, inout float len,
    float w,
    float lA, float lB, float lC, float lD, float lE)
{
    float lenX = max(abs(lD - lC), abs(lC - lB));
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) / (lenX + 1e-5), 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;

    float lenY = max(abs(lE - lC), abs(lC - lA));
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) / (lenY + 1e-5), 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

vec3 FsrEasuF(vec2 ip, vec4 con0, vec4 con1, vec4 con2, vec4 con3)
{
    vec2 pp = ip * con0.xy + con0.zw;
    vec2 fp = floor(pp);
    pp -= fp;

    vec2 p0 = fp * con1.xy + con1.zw;
    vec2 p1 = p0 + con2.xy;
    vec2 p2 = p0 + con2.zw;
    vec2 p3 = p0 + con3.xy;

    vec2 off = vec4(-0.5, 0.5, -0.5, 0.5) * con1.xxyy;

    // Sample the 12-tap pattern
    vec3 bC = readSrc(ivec2(p0 + off.xw)).rgb; float bL = Luminance(bC);
    vec3 cC = readSrc(ivec2(p0 + off.yw)).rgb; float cL = Luminance(cC);
    vec3 iC = readSrc(ivec2(p1 + off.xw)).rgb; float iL = Luminance(iC);
    vec3 jC = readSrc(ivec2(p1 + off.yw)).rgb; float jL = Luminance(jC);
    vec3 fC = readSrc(ivec2(p1 + off.yz)).rgb; float fL = Luminance(fC);
    vec3 eC = readSrc(ivec2(p1 + off.xz)).rgb; float eL = Luminance(eC);
    vec3 kC = readSrc(ivec2(p2 + off.xw)).rgb; float kL = Luminance(kC);
    vec3 lC = readSrc(ivec2(p2 + off.yw)).rgb; float lL = Luminance(lC);
    vec3 hC = readSrc(ivec2(p2 + off.yz)).rgb; float hL = Luminance(hC);
    vec3 gC = readSrc(ivec2(p2 + off.xz)).rgb; float gL = Luminance(gC);
    vec3 oC = readSrc(ivec2(p3 + off.yz)).rgb; float oL = Luminance(oC);
    vec3 nC = readSrc(ivec2(p3 + off.xz)).rgb; float nL = Luminance(nC);

    vec2 dir = vec2(0.0);
    float len = 0.0;

    FsrEasuSetF(dir, len, (1.0 - pp.x)*(1.0 - pp.y), bL, eL, fL, gL, jL);
    FsrEasuSetF(dir, len,        pp.x *(1.0 - pp.y), cL, fL, gL, hL, kL);
    FsrEasuSetF(dir, len, (1.0 - pp.x)*       pp.y , fL, iL, jL, kL, nL);
    FsrEasuSetF(dir, len,        pp.x *       pp.y , gL, jL, kL, lL, oL);

    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < 1.0/32768.0;
    dirR = inversesqrt(dirR);
    dirR = zro ? 1.0 : dirR;
    dir  = zro ? vec2(1.0, 0.0) : (dir * dirR);

    len = len * 0.5;
    len *= len;

    float stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
    vec2  len2 = vec2(1.0 + (stretch - 1.0)*len,
                      1.0 - 0.5*len);

    float lob = 0.5 - 0.29*len;
    float clp = 1.0/lob;

    vec3  aC = vec3(0.0);
    float aW = 0.0;

    FsrEasuTapF(aC, aW, vec2( 0,-1)-pp, dir, len2, lob, clp, bC);
    FsrEasuTapF(aC, aW, vec2( 1,-1)-pp, dir, len2, lob, clp, cC);
    FsrEasuTapF(aC, aW, vec2(-1, 1)-pp, dir, len2, lob, clp, iC);
    FsrEasuTapF(aC, aW, vec2( 0, 1)-pp, dir, len2, lob, clp, jC);
    FsrEasuTapF(aC, aW, vec2( 0, 0)-pp, dir, len2, lob, clp, fC);
    FsrEasuTapF(aC, aW, vec2(-1, 0)-pp, dir, len2, lob, clp, eC);
    FsrEasuTapF(aC, aW, vec2( 1, 1)-pp, dir, len2, lob, clp, kC);
    FsrEasuTapF(aC, aW, vec2( 2, 1)-pp, dir, len2, lob, clp, lC);
    FsrEasuTapF(aC, aW, vec2( 2, 0)-pp, dir, len2, lob, clp, hC);
    FsrEasuTapF(aC, aW, vec2( 1, 0)-pp, dir, len2, lob, clp, gC);
    FsrEasuTapF(aC, aW, vec2( 1, 2)-pp, dir, len2, lob, clp, oC);
    FsrEasuTapF(aC, aW, vec2( 0, 2)-pp, dir, len2, lob, clp, nC);

    vec3 min4 = min(min(fC, gC), min(jC, kC));
    vec3 max4 = max(max(fC, gC), max(jC, kC));
    return min(max4, max(min4, aC / aW));
}

// Bilinear fallback
vec4 bilinearUpscale(ivec2 srcCoord) {
    vec2 uv = (vec2(srcCoord) + 0.5) / vec2(srcWidth, srcHeight);
    vec2 dstSize = vec2(dstWidth, dstHeight);
    ivec2 dstBase = ivec2(uv * dstSize);

    vec4 c00 = readSrc(dstBase);
    vec4 c10 = readSrc(dstBase + ivec2(1, 0));
    vec4 c01 = readSrc(dstBase + ivec2(0, 1));
    vec4 c11 = readSrc(dstBase + ivec2(1, 1));

    vec2 frac = fract(uv * dstSize);
    vec4 c0 = mix(c00, c10, frac.x);
    vec4 c1 = mix(c01, c11, frac.x);
    return mix(c0, c1, frac.y);
}

void main() {
    ivec2 dstCoord = ivec2(gl_GlobalInvocationID.xy);
    if (dstCoord.x >= dstWidth || dstCoord.y >= dstHeight)
        return;

    // If AI upscaling is disabled, use bilinear
    if (params.enableAIUpscaling == 0) {
        // Map destination back to source coordinates
        vec2 srcUV = (vec2(dstCoord) + 0.5) / vec2(dstWidth, dstHeight);
        ivec2 srcCoord = ivec2(srcUV * vec2(srcWidth, srcHeight));
        srcCoord = clamp(srcCoord, ivec2(0, 0), ivec2(srcWidth - 1, srcHeight - 1));
        vec4 result = bilinearUpscale(srcCoord);
        imageStore(dstImg, dstCoord, result);
        return;
    }

    // FSR EASU 4x path - map destination to source space
    vec2 ip = (vec2(dstCoord) + 0.5) * 0.25;  // inverse of 4x scale

    vec2 texSize = vec2(srcWidth, srcHeight);
    vec2 outSize = vec2(dstWidth, dstHeight);

    vec4 con0, con1, con2, con3;
    FsrEasuCon(con0, con1, con2, con3, texSize, texSize, outSize);

    vec3 rgb = FsrEasuF(ip, con0, con1, con2, con3);
    imageStore(dstImg, dstCoord, vec4(rgb, 1.0));
}