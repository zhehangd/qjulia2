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

#include "core/shape/julia3d.h"

#include <vector>
#include <memory>

#include "core/arg_parse.h"
#include "core/vector.h"
#include "core/shape.h"
#include "core/algorithm.h"

namespace qjulia {

#ifdef WITH_CUDA

struct Julia3DData {
  Quaternion constant;
  Float precision;
  Float max_magnitude;
  Float bounding_radius;
  int max_iterations;
};

KERNEL void UpdateJulia3DShape(Entity *dst_b, Julia3DData params) {
  auto *dst = static_cast<Julia3DShape*>(dst_b);
  dst->SetConstant(params.constant);
  dst->SetPrecision(params.precision);
  dst->SetEscapeMagnitude(params.max_magnitude);
  dst->SetBoundingRadius(params.bounding_radius);
  dst->SetMaxInterations(params.max_iterations);
}

void Julia3DShape::UpdateDevice(Entity *device_ptr) const {
  Julia3DData params;
  params.constant = kernel.GetConstant();
  params.precision = kernel.GetPrecision();
  params.max_magnitude = kernel.GetEscapeMagnitude();
  params.bounding_radius = kernel.GetBoundingRadius();
  params.max_iterations = kernel.GetMaxInterations();
  UpdateJulia3DShape<<<1, 1>>>(device_ptr, params);
}

#else

void Julia3DShape::UpdateDevice(Entity*) const {}

#endif

CPU_AND_CUDA void Julia3DShape::UsePreset(int i) {
  Quaternion constant;
  switch (i) {
    case 0: constant = {-1, 0.2, 0, 0}; break;
    case 1: constant = {-0.2, 0.8, 0, 0}; break;
    case 2: constant = {-0.125,-0.256,0.847,0.0895}; break;
    default:;
  }
  SetConstant(constant);
}

void Julia3DShape::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "SetConstant") {
    Vector3f v;
    ParseArg(args[1], v);
    Quaternion q;
    for (int i = 0; i < 3; ++i) {q[i] = v[i];}
    SetConstant(q);
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
