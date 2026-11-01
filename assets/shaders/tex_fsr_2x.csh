// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2024- PPSSPP Project.
// FSR 1.0 EASU (Edge-Adaptive Spatial Upscaling) compute shader for texture upscaling (2x)
// Adapted from AMD's FSR 1.0 reference implementation (MIT License)

#define srcWidth params.width
#define srcHeight params.height
#define dstWidth (params.width * 2)
#define dstHeight (params.height * 2)

// FSR EASU constants
const float FSR_EPS = 0.0001;

// Read source pixel (clamped)
vec4 readSrc(ivec2 p) {
    p = clamp(p, ivec2(0, 0), ivec2(srcWidth - 1, srcHeight - 1));
    return readColorf(uvec2(p));
}

vec4 readSrc(vec2 uv) {
    ivec2 p = ivec2(floor(uv * vec2(srcWidth, srcHeight)));
    return readSrc(p);
}

float Luminance(vec3 rgb) {
	// AMD's approximate luma multiplied by two.
	return rgb.b * 0.5 + (rgb.r * 0.5 + rgb.g);
}

float FsrApproxLoRcpF1(float x) {
	return uintBitsToFloat(0x7ef07ebbu - floatBitsToUint(x));
}

float FsrApproxLoRsqF1(float x) {
	return uintBitsToFloat(0x5f347d74u - (floatBitsToUint(x) >> 1u));
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
    inout vec4 aC, inout float aW,
    vec2 off,
    vec2 dir, vec2 len,
    float lob, float clp,
    vec4 c)
{
    vec2 v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x))) * len;
    float d2 = min(dot(v, v), clp);
    float wB = (2.0 / 5.0) * d2 + (-1.0);
    float wA = lob * d2 + (-1.0);
    wB *= wB; wA *= wA;
    wB = (25.0 / 16.0) * wB + (-(25.0 / 16.0 - 1.0));
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
    lenX = FsrApproxLoRcpF1(lenX);
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;

    float lenY = max(abs(lE - lC), abs(lC - lA));
    lenY = FsrApproxLoRcpF1(lenY);
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

vec4 FsrEasuF(vec2 ip, vec4 con0, vec4 con1, vec4 con2, vec4 con3)
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
    vec4 bC = readSrc(p0 + off.xw); float bL = Luminance(bC.rgb);
    vec4 cC = readSrc(p0 + off.yw); float cL = Luminance(cC.rgb);
    vec4 iC = readSrc(p1 + off.xw); float iL = Luminance(iC.rgb);
    vec4 jC = readSrc(p1 + off.yw); float jL = Luminance(jC.rgb);
    vec4 fC = readSrc(p1 + off.yz); float fL = Luminance(fC.rgb);
    vec4 eC = readSrc(p1 + off.xz); float eL = Luminance(eC.rgb);
    vec4 kC = readSrc(p2 + off.xw); float kL = Luminance(kC.rgb);
    vec4 lC = readSrc(p2 + off.yw); float lL = Luminance(lC.rgb);
    vec4 hC = readSrc(p2 + off.yz); float hL = Luminance(hC.rgb);
    vec4 gC = readSrc(p2 + off.xz); float gL = Luminance(gC.rgb);
    vec4 oC = readSrc(p3 + off.yz); float oL = Luminance(oC.rgb);
    vec4 nC = readSrc(p3 + off.xz); float nL = Luminance(nC.rgb);

    vec2 dir = vec2(0.0);
    float len = 0.0;

    FsrEasuSetF(dir, len, (1.0 - pp.x)*(1.0 - pp.y), bL, eL, fL, gL, jL);
    FsrEasuSetF(dir, len,        pp.x *(1.0 - pp.y), cL, fL, gL, hL, kL);
    FsrEasuSetF(dir, len, (1.0 - pp.x)*       pp.y , fL, iL, jL, kL, nL);
    FsrEasuSetF(dir, len,        pp.x *       pp.y , gL, jL, kL, lL, oL);

    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < 1.0/32768.0;
    dirR = FsrApproxLoRsqF1(dirR);
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;

    len = len * 0.5;
    len *= len;

    float stretch = (dir.x * dir.x + dir.y * dir.y) * FsrApproxLoRcpF1(max(abs(dir.x), abs(dir.y)));
    vec2  len2 = vec2(1.0 + (stretch - 1.0)*len,
                      1.0 - 0.5*len);

    float lob = 0.5 + (0.25 - 0.04 - 0.5) * len;
    float clp = FsrApproxLoRcpF1(lob);

    vec4  aC = vec4(0.0);
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

    vec4 min4 = min(min(fC, gC), min(jC, kC));
    vec4 max4 = max(max(fC, gC), max(jC, kC));
    return min(max4, max(min4, aC * (1.0 / aW)));
}

// Bilinear fallback
vec4 bilinearUpscale(ivec2 dstCoord) {
    vec2 srcPos = (vec2(dstCoord) + 0.5) * vec2(srcWidth, srcHeight) / vec2(dstWidth, dstHeight) - 0.5;
    ivec2 srcBase = ivec2(floor(srcPos));
    vec2 frac = fract(srcPos);

    vec4 c00 = readSrc(srcBase);
    vec4 c10 = readSrc(srcBase + ivec2(1, 0));
    vec4 c01 = readSrc(srcBase + ivec2(0, 1));
    vec4 c11 = readSrc(srcBase + ivec2(1, 1));

    vec4 c0 = mix(c00, c10, frac.x);
    vec4 c1 = mix(c01, c11, frac.x);
    return mix(c0, c1, frac.y);
}

void applyScaling(uvec2 xy) {
    vec2 texSize = vec2(srcWidth, srcHeight);
    vec2 outSize = vec2(dstWidth, dstHeight);
    vec4 con0, con1, con2, con3;
    if (params.enableAIUpscaling != 0) {
        FsrEasuCon(con0, con1, con2, con3, texSize, texSize, outSize);
    }

    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            ivec2 dstCoord = ivec2(xy * 2u + uvec2(x, y));

            // If upscaling is disabled, use the existing bilinear fallback.
            if (params.enableAIUpscaling == 0) {
                writeColorf(dstCoord, bilinearUpscale(dstCoord));
                continue;
            }

            // EASU takes the integer position in the output image. con0 applies the scale.
            vec4 rgba = FsrEasuF(vec2(dstCoord), con0, con1, con2, con3);
            writeColorf(dstCoord, rgba);
        }
    }
}
