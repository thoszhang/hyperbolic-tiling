#version 300 es

precision highp float;

uniform vec2 u_resolution;
uniform mat4 u_matA;
uniform mat4 u_matB;
uniform mat4 u_matC;
uniform mat4 u_invHalfPlA;
uniform mat4 u_invHalfPlB;
uniform mat4 u_invHalfPlC;
uniform mat4 u_invHalfPl1;
uniform mat4 u_invHalfPl2;
uniform mat4 u_invHalfPl3;
uniform int u_mode;
uniform int u_showTriangles;

out vec4 outputColor;

mat2 complex(float re, float im) {
  return mat2(re, im, -im, re);
}

mat2x4 homogenize(mat2 z) {
  return mat2x4(z[0], 1.0, 0.0, z[1], 0.0, 1.0);
}

mat2 dehomogenize(mat2x4 p) {
  return mat2(p[0].xy, p[1].xy) * inverse(mat2(p[0].zw, p[1].zw));
}

mat2x4 conj(mat2x4 p) {
  return mat2x4(
    p[0].x, p[1].x, p[0].z, p[1].z,
    p[0].y, p[1].y, p[0].w, p[1].w
  );
}

bool inHalfPlane(mat4 invPlane, mat2x4 p) {
  return dehomogenize(invPlane * p)[0].y >= 0.0;
}

int x_yzRegion(mat2x4 p) {
  if (inHalfPlane(u_invHalfPl1, p)) {
    return 0;
  }
  return 1;
}

int xy_zRegion(mat2x4 p) {
  if (!inHalfPlane(u_invHalfPl1, p)) {
    return 1;
  }
  if (!inHalfPlane(u_invHalfPl2, p)) {
    return 2;
  }
  return 0;
}

int pqr_Region(mat2x4 p) {
  bool ins1 = inHalfPlane(u_invHalfPl1, p);
  bool ins2 = inHalfPlane(u_invHalfPl2, p);
  bool ins3 = inHalfPlane(u_invHalfPl3, p);

  if (ins2 && !ins3) {
    return 1;
  }
  if (ins3 && !ins1) {
    return 2;
  }
  return 0;
}

void main() {
  mat2 z = complex(
    2.0 * gl_FragCoord.x / u_resolution.x - 1.0,
    1.0 - 2.0 * gl_FragCoord.y / u_resolution.y
  );

  if (determinant(z) > 1.0) {
    outputColor = vec4(1, 1, 1, 1);
    return;
  }

  mat2x4 p = homogenize(z);
  bool insA = inHalfPlane(u_invHalfPlA, p);
  bool insB = inHalfPlane(u_invHalfPlB, p);
  bool insC = inHalfPlane(u_invHalfPlC, p);
  int numRefls = 0;

  while ((!insA || !insB || !insC) && numRefls < 50) {
    if (!insA) {
      p = u_matA * conj(p);
    } else if (!insB) {
      p = u_matB * conj(p);
    } else {
      p = u_matC * conj(p);
    }
    insA = inHalfPlane(u_invHalfPlA, p);
    insB = inHalfPlane(u_invHalfPlB, p);
    insC = inHalfPlane(u_invHalfPlC, p);
    numRefls += 1;
  }

  if (!insA || !insB || !insC) {
    outputColor = vec4(.5, 0.5, 0.5, 1);
    return;
  }

  int triangleRegion = numRefls % 2;
  float val;
  if (u_mode == 0) {
    val = float(triangleRegion);
  } else {
    int region;
    if (u_mode == 1) {
      region = x_yzRegion(p);
    } else if (u_mode == 2) {
      region = xy_zRegion(p);
    } else if (u_mode == 3) {
      region = pqr_Region(p);
    }
    val = (float(region) + 1.0) / 4.0;
    if (u_showTriangles == 1) {
      val += (2.0 * float(triangleRegion) - 1.0) / 16.0;
    }
  }

  outputColor = vec4(val, val, val, 1);
}