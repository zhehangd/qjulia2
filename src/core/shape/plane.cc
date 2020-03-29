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

#include "core/shape/plane.h"

#include "math.h"

#include <vector>
#include <memory>

#include "core/arg_parse.h"
#include "core/algorithm.h"
#include "core/shape.h"
#include "core/vector.h"

namespace qjulia {

CPU_AND_CUDA  PlaneShape::PlaneShape(void) : PlaneShape({}, {0, 0, 1}) {
}

CPU_AND_CUDA  PlaneShape::PlaneShape(Vector3f position, Vector3f orientation) {
  SetPositionAndNormal(position, orientation);
}

CPU_AND_CUDA void PlaneShape::SetPositionAndNormal(Vector3f position, Vector3f orientation) {
  normal = Normalize(orientation);
  offset = - Dot(position, normal);
}

CPU_AND_CUDA Intersection PlaneShape::Intersect(const Ray &ray) const {
  Intersection isect;
  Float num = - (Dot(normal, ray.start) + offset);
  Float den = Dot(normal, ray.dir);
  Float dist;
  if (isinf(num) || isinf(den)) {
    dist = signbit(num) == signbit(den) ? kInf : kNInf;
  } else {
    dist = num / den;
  }
  if (dist >= 0) {
    isect.good = true;
    isect.dist = dist;
    isect.position = ray.start + ray.dir * dist;
    isect.normal = normal;
  }
  return isect;
}

#ifdef WITH_CUDA

struct PlaneData {
  Vector3f normal;
  Float offset;
};

KERNEL void UpdatePlaneShape(Entity *dst_b, PlaneData params) {
  auto *dst = static_cast<PlaneShape*>(dst_b);
  dst->normal = params.normal;
  dst->offset = params.offset;
}

void PlaneShape::UpdateDevice(Entity *device_ptr) const {
  PlaneData params;
  params.normal = normal;
  params.offset = offset;
  UpdatePlaneShape<<<1, 1>>>(device_ptr, params);
}

#else

void PlaneShape::UpdateDevice(Entity*) const {}

#endif

void PlaneShape::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "Set") {
    Vector3f v2[2];
    ParseArg(args[1], v2[0]);
    ParseArg(args[2], v2[1]);
    SetPositionAndNormal(v2[0], v2[1]);
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
