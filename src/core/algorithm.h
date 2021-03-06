/*

MIT License

Copyright (c) 2019 Zhehang Ding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef QJULIA_ALGORITHM_H_
#define QJULIA_ALGORITHM_H_

/** \file Common algorithms
*/

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "base.h"
#include "ray.h"
#include "vector.h"

namespace qjulia {
  
CPU_AND_CUDA inline Float Deg2Rad(Float a) {
  return a * 3.14159265 / 180;
}

CPU_AND_CUDA inline Float Rad2Deg(Float a) {
  return a * 180 / 3.14159265;
}

/** \brief Solve a quadratic equation

Solve a*x^2 + b*x + c = 0 for its real roots. If the equation has real roots,
they are solved in tl and tg and true is returned. Otherwise tl are tg are
not changed and false is returned.
*/
CPU_AND_CUDA inline bool SolveQuadratic(
    Float a, Float b, Float c, Float *tl, Float *tg) {
  // TODO This is a naive implementation which is not stable
  // when a is close to 0. Improvement is needed.
  Float d = b * b - 4 * a * c;
  if (d < 0) {
    return false;
  } else {
    d = std::sqrt(d);
    *tl = (-b + d) / (2 * a);
    *tg = (-b - d) / (2 * a);
    if (*tl > *tg) {
      Float t = *tl;
      *tl = *tg;
      *tg = t;
    }
    return true;
  }
}

CPU_AND_CUDA inline bool IntersectSphere(
    const Vector3f start, const Vector3f dir,
    Float r, Float *tl, Float *tg) {
  Float a = dir.Norm2();
  Float b = 2 * Dot(start, dir);
  Float c = start.Norm2() - r * r;
  bool has_root = SolveQuadratic(a, b, c, tl, tg);
  return has_root;
}

// This function produces a basis of the vector space orthogonal to a unit
// vector 'vec'. The number of basis vectors equals the dimension minus one.
// NOTE: 'vec' must has magnitude one otherwise the result would be incorrect.
// The magnitude of the basis vectors are not necessarily one.
template<typename Float, int C>
std::vector<Vec_<Float, C>> ProduceOrthogonalBasis(const Vec_<Float, C> &vec) {
  // Pick the dimension corresponding to the maximum coordinate of 'vec'.
  const int dim = C;
  int max_dim = 0;
  Float max_v = std::abs(vec[0]);
  for (int i = 1; i < dim; ++i) {
    Float v = std::abs(vec[i]);
    if (v > max_v) {
      max_dim = i;
      max_v = v;
    }
  }
  
  // Generate basis vectors.
  std::vector<Vector3f> basis;
  basis.reserve(dim - 1);
  for (int i = 0; i < dim; ++i) {
    if (i == max_dim) {continue;}
    Vector3f basis_vec;
    basis_vec[i] = 1.0f;
    basis_vec = basis_vec - vec * Dot(vec, basis_vec);
    basis.push_back(basis_vec);
  }
  return basis;
}

inline Vector3f Spherical2CartesianCoords(Vector3f src) {
  Float dist = src[2];
  Float vdist = dist * std::sin(Deg2Rad(src[1]));
  Float hdist = dist * std::cos(Deg2Rad(src[1]));
  Float x = hdist * std::sin(Deg2Rad(src[0]));
  Float z = hdist * std::cos(Deg2Rad(src[0]));
  return {x, vdist, z};
}

inline Vector3f Cartesian2SphericalCoords(Vector3f src) {
  Float dist = std::sqrt(src[0] * src[0] + src[1] * src[1] + src[2] * src[2]);
  if (dist == 0) {return {0, 0, 0};}
  Float azi = Rad2Deg(std::atan2(src[0], src[2]));
  Float alt = Rad2Deg(std::asin(src[1] / dist));
  return {azi, alt, dist};
}

CPU_AND_CUDA inline std::uint8_t ClipTo8Bit(Float v) {
  return (std::uint8_t)std::round(
    std::fmin((Float)255.0, 
    std::fmax((Float)0.0, v)));
}

CPU_AND_CUDA inline Pixel ClipTo8Bit(Vector3f v) {
  return Pixel(ClipTo8Bit(v[0]), ClipTo8Bit(v[1]), ClipTo8Bit(v[2]));
}

}

#endif
