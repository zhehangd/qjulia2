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
  bool cross_section = false;
  Float uv_black;
  Float uv_white;
};

KERNEL void UpdateJulia3DShape(Entity *dst_b, Julia3DData params) {
  auto *dst = static_cast<Julia3DShape*>(dst_b);
  dst->SetConstant(params.constant);
  dst->SetPrecision(params.precision);
  dst->SetEscapeMagnitude(params.max_magnitude);
  dst->SetBoundingRadius(params.bounding_radius);
  dst->SetMaxInterations(params.max_iterations);
  dst->SetCrossSectionFlag(params.cross_section);
  dst->SetUVBlack(params.uv_black);
  dst->SetUVWhite(params.uv_white);
}

void Julia3DShape::UpdateDevice(Entity *device_ptr) const {
  Julia3DData params;
  params.constant = kernel.GetConstant();
  params.precision = kernel.GetPrecision();
  params.max_magnitude = kernel.GetEscapeMagnitude();
  params.bounding_radius = kernel.GetBoundingRadius();
  params.max_iterations = kernel.GetMaxInterations();
  params.cross_section = kernel.GetCrossSectionFlag();
  params.uv_black = kernel.GetUVBlack();
  params.uv_white = kernel.GetUVWhite();
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
    Quaternion q;
    ParseArg(args[1], q);
    SetConstant(q);
  } else if (args[0] == "SetPrecision") {
    Float precision;
    ParseArg(args[1], precision);
    SetPrecision(precision);
  } else if (args[0] == "SetMaxInterations") {
    int max_iterations;
    ParseArg(args[1], max_iterations);
    SetMaxInterations(max_iterations);
  } else if (args[0] == "SetEscapeMagnitude") {
    Float mag;
    ParseArg(args[1], mag);
    SetEscapeMagnitude(mag);
  } else if (args[0] == "SetBoundingRadius") {
    Float radius;
    ParseArg(args[1], radius);
    SetBoundingRadius(radius);
  } else if (args[0] == "SetCrossSectionFlag") {
    bool flag;
    ParseArg(args[1], flag);
    SetCrossSectionFlag(flag);
  } else if (args[0] == "SetUVBlack") {
    Float val;
    ParseArg(args[1], val);
    SetUVBlack(val);
  }  else if (args[0] == "SetUVWhite") {
    Float val;
    ParseArg(args[1], val);
    SetUVWhite(val);
  }  else {
    throw UnknownCommand(args[0]);
  }
}

void Julia3DShape::Save(SceneBuilder *build, FnSaveArgs fn_write) const {
  (void)build;
  fn_write({"SetConstant", ToString(GetConstant())});
  fn_write({"SetPrecision", ToString(GetPrecision())});
  fn_write({"SetMaxInterations", ToString(GetMaxInterations())});
  fn_write({"SetEscapeMagnitude", ToString(GetEscapeMagnitude())});
  fn_write({"SetBoundingRadius", ToString(GetBoundingRadius())});
  fn_write({"SetCrossSectionFlag", ToString(GetCrossSectionFlag())});
  fn_write({"SetUVBlack", ToString(GetUVBlack())});
  fn_write({"SetUVWhite", ToString(GetUVWhite())});
}

}
