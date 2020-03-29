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

#include "core/material.h"

#include "core/arg_parse.h"
#include "core/scene_descr.h"

namespace qjulia {
  
#ifdef WITH_CUDA

struct MaterialData {
  Vector3f diffuse;
  Float ks;
  Float ps;
  Float reflection;
};

KERNEL void UpdatePointLight(Entity *dst_b, MaterialData params) {
  auto *dst = static_cast<Material*>(dst_b);
  dst->diffuse = params.diffuse;
  dst->ks = params.ks;
  dst->ps = params.ps;
  dst->reflection = params.reflection;
}

void Material::UpdateDevice(Entity *device_ptr) const {
  MaterialData params;
  params.diffuse = diffuse;
  params.ks = ks;
  params.ps = ps;
  params.reflection = reflection;
  UpdatePointLight<<<1, 1>>>(device_ptr, params);
}

#endif

void Material::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "SetDiffuse") {
    ParseArg(args[1], diffuse);
  } else if (args[0] == "SetReflection") {
    ParseArg(args[1], reflection);
  } else if (args[0] == "SetSpecular") {
    ParseArg(args[1], ks);
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
