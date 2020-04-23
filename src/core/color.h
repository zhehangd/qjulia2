#ifndef QJULIA_COLOR_H_
#define QJULIA_COLOR_H_

#include "vector.h"

#include <cmath>

namespace qjulia {

CPU_AND_CUDA inline Vector3f RGB2XYZ(Vector3f src) {
  Vector3f dst;
  dst[0] = 0.412453 * src[0] + 0.357580 * src[1] + 0.180423 * src[2];
  dst[1] = 0.212671 * src[0] + 0.715160 * src[1] + 0.072169 * src[2];
  dst[2] = 0.019334 * src[0] + 0.119193 * src[1] + 0.950227 * src[2];
  return dst;
}

CPU_AND_CUDA inline Vector3f XYZ2RGB(Vector3f src) {
  Vector3f dst;
  dst[0] =  3.240479 * src[0] + -1.53715  * src[1] + -0.498535 * src[2];
  dst[1] = -0.969256 * src[0] + 1.875991  * src[1] +  0.041556 * src[2];
  dst[2] =  0.055648 * src[0] + -0.204043 * src[1] +  1.057311 * src[2];
  return dst;
}

CPU_AND_CUDA inline Vector3f XYZ2Lab(Vector3f src) {
  constexpr Float d = 6.0 / 29.0;
  constexpr Float d3 = d * d * d; // 0.00856
  constexpr Float a = 1 / (3 * d * d); // 7.787
  src[0] = src[0] / 0.950456;
  src[2] = src[2] / 1.088754;
  auto fn = [&](Float t) {
    return t > d3 ? std::cbrt(t) : a * t + 16.0 / 116.0;
  };
  Float fx = fn(src[0]);
  Float fy = fn(src[1]);
  Float fz = fn(src[2]);
  Vector3f dst;
  dst[0] = (116 * fy - 16);
  dst[1] = 500 * (fx - fy);
  dst[2] = 200 * (fy - fz);
  dst[0] /= 100;
  dst[1] /= 127;
  dst[2] /= 127;
  return dst;
}

CPU_AND_CUDA inline Vector3f Lab2XYZ(Vector3f src) {
  src[0] *= 100;
  src[1] *= 127;
  src[2] *= 127;
  constexpr Float d = 6.0 / 29.0;
  Float fy = (src[0] + 16) / 116;
  Float fx = fy + src[1] / 500;
  Float fz = fy - src[2] / 200;
  auto fn = [&](Float t) {
    return t > d ? (t * t * t) : (t - 16.0 / 116.0) * (3 * d * d);
  };
  Vector3f dst;
  dst[0] = fn(fx) * 0.950456;
  dst[1] = fn(fy);
  dst[2] = fn(fz) * 1.088754;
  return dst;
}

CPU_AND_CUDA inline Vector3f Lab2LCH(Vector3f src) {
  Float l = src[0];
  Float c = std::sqrt(src[1] * src[1] + src[2] * src[2]);
  Float h = std::atan2(src[2], src[1]) * 180 / 3.14159265;
  return {l, c, h};
}

CPU_AND_CUDA inline Vector3f LCH2Lab(Vector3f src) {
  float a = src[2] * 3.14159265 / 180;
  return {src[0], src[1] * std::cos(a), src[1] * std::sin(a)};
}

CPU_AND_CUDA inline Vector3f RGB2Lab(Vector3f src) {
  return XYZ2Lab(RGB2XYZ(src));
}

CPU_AND_CUDA inline Vector3f Lab2RGB(Vector3f src) {
  return XYZ2RGB(Lab2XYZ(src));
}

CPU_AND_CUDA inline Vector3f RGB2LCH(Vector3f src) {
  return Lab2LCH(RGB2Lab(src));
}

CPU_AND_CUDA inline Vector3f LCH2RGB(Vector3f src) {
  return Lab2RGB(LCH2Lab(src));
}

}

#endif
