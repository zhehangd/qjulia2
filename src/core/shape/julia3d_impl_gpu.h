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

#ifndef QJULIA_JULIA3D_IMPL_GPU_H_
#define QJULIA_JULIA3D_IMPL_GPU_H_

#include <vector>

#include "core/array2d.h"
#include "core/ray.h"
#include "core/intersection.h"
#include "core/vector.h"

namespace qjulia {

void Julia3DIntersectGPU(
  const Ray &ray, Intersection &isect,
  Quaternion julia_constant, int max_iterations,
  Float max_magnitude, Float bounding_radius);

void Julia3DIntersectGPU(
  const Array2D<Ray> &rays, Array2D<Intersection> &isects,
  Quaternion julia_constant, int max_iterations,
  Float max_magnitude, Float bounding_radius);

}

#endif
