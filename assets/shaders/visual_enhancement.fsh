// Copyright (c) 2012- PPSSPP Project.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

uniform sampler2D sampler0;
uniform vec2 u_texelDelta;
uniform vec4 u_setting;

varying vec2 v_texcoord0;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 EnhanceColor(vec3 color) {
	// Add more saturation to muted colors than to already vivid colors.
	float saturation = max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
	float luma = dot(color, LUMA);
	color = vec3(luma) + (color - vec3(luma)) * (1.0 + u_setting.x * (1.0 - saturation));

	// Apply a display-referred contrast adjustment around middle gray.
	color = (color - vec3(0.5)) * u_setting.y + vec3(0.5);

	// Gently lift deep shadows and roll off highlights while retaining their detail.
	vec3 shadows = max(vec3(1.0) - color, vec3(0.0));
	vec3 highlights = max(color - vec3(0.72), vec3(0.0)) / 0.28;
	color += vec3(u_setting.z * 0.035) * shadows * shadows * shadows;
	color -= vec3(u_setting.z * 0.045) * highlights * highlights;
	return clamp(color, 0.0, 1.0);
}

void main() {
	vec2 uv = v_texcoord0;
	vec4 source = texture2D(sampler0, uv);
	vec3 center = source.rgb;
	vec3 north = texture2D(sampler0, uv - vec2(0.0, u_texelDelta.y)).rgb;
	vec3 south = texture2D(sampler0, uv + vec2(0.0, u_texelDelta.y)).rgb;
	vec3 west = texture2D(sampler0, uv - vec2(u_texelDelta.x, 0.0)).rgb;
	vec3 east = texture2D(sampler0, uv + vec2(u_texelDelta.x, 0.0)).rgb;

	// Limit sharpening to the range already present in the local neighborhood to reduce halos.
	vec3 neighborhoodMin = min(center, min(min(north, south), min(west, east)));
	vec3 neighborhoodMax = max(center, max(max(north, south), max(west, east)));
	vec3 detail = center - (north + south + west + east) * 0.25;
	float detailLevel = max(max(abs(detail.r), abs(detail.g)), abs(detail.b));
	float detailGate = smoothstep(0.008, 0.04, detailLevel);
	vec3 sharpened = clamp(center + detail * (1.4 * u_setting.w * detailGate), neighborhoodMin, neighborhoodMax);

	gl_FragColor = vec4(EnhanceColor(sharpened), source.a);
}
