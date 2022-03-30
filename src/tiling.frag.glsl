#version 300 es

precision highp float;

uniform vec2 u_resolution;
uniform mat4 u_matA;
uniform mat4 u_matB;
uniform mat4 u_matC;
uniform mat4 u_invHalfPlA;
uniform mat4 u_invHalfPlB;
uniform mat4 u_invHalfPlC;
uniform mat4 u_mat1;
uniform mat4 u_mat2;
uniform mat4 u_mat3;
uniform mat4 u_invHalfPl1;
uniform mat4 u_invHalfPl2;
uniform mat4 u_invHalfPl3;
uniform int u_mode;
uniform int u_showTriangles;
uniform int u_showEdges;
uniform int u_colorPolygons;

out vec4 outputColor;

const float thickness = 0.05;

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

float approxDistToHalfPlane(mat4 invPlane, mat2x4 p) {
  return dehomogenize(invPlane * p)[0].y;
}

bool inHalfPlane(mat4 invPlane, mat2x4 p) {
  return approxDistToHalfPlane(invPlane, p) >= 0.0;
}

float distToLine2(mat4 refl, mat2x4 p) {
  mat2 u = dehomogenize(refl * conj(p));
  mat2 v = dehomogenize(p);
  float delta = 2.0 * determinant(u - v) / ((1.0 - determinant(u)) * (1.0 - determinant(v)));
  return acosh(1.0 + delta);
}

bool inTrianglesBoundary(mat2x4 p) {
  bool inA = distToLine2(u_matA, p) < thickness;
  bool inB = distToLine2(u_matB, p) < thickness;
  bool inC = distToLine2(u_matC, p) < thickness;
  return inA || inB || inC;
}

int x_yzRegion(mat2x4 p) {
  if (inHalfPlane(u_invHalfPl1, p)) {
    return 0;
  }
  return 1;
}

bool inx_yzBoundary(mat2x4 p) {
  return distToLine2(u_mat1, p) < thickness;
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

bool inxy_zBoundary(mat2x4 p) {
  bool in1 = distToLine2(u_mat1, p) < thickness && inHalfPlane(u_invHalfPl2, p);
  bool in2 = distToLine2(u_mat2, p) < thickness && inHalfPlane(u_invHalfPl1, p);
  return in1 || in2;
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

bool inpqr_Boundary(mat2x4 p) {
  bool ins1 = inHalfPlane(u_invHalfPl1, p);
  bool ins2 = inHalfPlane(u_invHalfPl2, p);
  bool ins3 = inHalfPlane(u_invHalfPl3, p);

  bool in1 = distToLine2(u_mat1, p) < thickness && !(ins2 && !ins3);
  bool in2 = distToLine2(u_mat2, p) < thickness && !(ins3 && !ins1);
  bool in3 = distToLine2(u_mat3, p) < thickness && !(ins1 && !ins2);
  return in1 || in2 || in3;
}

void main() {
  mat2 z = complex(
    2.0 * gl_FragCoord.x / u_resolution.x - 1.0,
    2.0 * gl_FragCoord.y / u_resolution.y - 1.0
  );

  if (determinant(z) > 1.0) {
    outputColor = vec4(1, 1, 1, 1);
    return;
  }

  mat2x4 p = homogenize(z);
  float approxDistA = approxDistToHalfPlane(u_invHalfPlA, p);
  float approxDistB = approxDistToHalfPlane(u_invHalfPlB, p);
  float approxDistC = approxDistToHalfPlane(u_invHalfPlC, p);
  int numRefls = 0;

  while ((approxDistA < 0.0 || approxDistB < 0.0 || approxDistC < 0.0) && numRefls < 50) {
    if (approxDistA <= approxDistB && approxDistA <= approxDistC) {
      p = u_matA * conj(p);
    } else if (approxDistB <= approxDistC && approxDistB <= approxDistA) {
      p = u_matB * conj(p);
    } else {
      p = u_matC * conj(p);
    }
    approxDistA = approxDistToHalfPlane(u_invHalfPlA, p);
    approxDistB = approxDistToHalfPlane(u_invHalfPlB, p);
    approxDistC = approxDistToHalfPlane(u_invHalfPlC, p);
    numRefls += 1;
  }

  if (approxDistA < 0.0 || approxDistB < 0.0 || approxDistC < 0.0) {
    outputColor = vec4(0.5, 0.5, 0.5, 1);
    return;
  }

  int triangleRegion = numRefls % 2;
  float val = 0.5;

  if (u_colorPolygons == 1) {
    int region;
    if (u_mode == 0) {
      region = 1;
    } else if (u_mode == 1) {
      region = x_yzRegion(p);
    } else if (u_mode == 2) {
      region = xy_zRegion(p);
    } else if (u_mode == 3) {
      region = pqr_Region(p);
    }
    val += 0.25 * (float(region) - 1.0);
  }

  if (u_showTriangles == 1) {
    if (u_mode == 0) {
      val += (2.0 * float(triangleRegion) - 1.0) / 8.0;
    } else {
      val += (2.0 * float(triangleRegion) - 1.0) / 16.0;
    }
  }

  if (u_showEdges == 1) {
    if (u_mode == 0) {
      if (inTrianglesBoundary(p)) {
        val = 0.0;
      }
    } else if (u_mode == 1) {
      if (inx_yzBoundary(p)) {
        val = 0.0;
      }
    } else if (u_mode == 2) {
      if (inxy_zBoundary(p)) {
        val = 0.0;
      }
    } else if (u_mode == 3) {
      if (inpqr_Boundary(p)) {
        val = 0.0;
      }
    }
  }

  outputColor = vec4(val, val, val, 1);
}