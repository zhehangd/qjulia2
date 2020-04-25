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

#include "simple.h"

#include "core/arg_parse.h"

namespace qjulia {

CPU_AND_CUDA LightRay SunLight::Li(const Point3f &p) const {
  (void)p;
  LightRay lray;
  lray.dist = kInf;
  lray.wi = - Normalize(orientation);
  lray.spectrum = intensity;
  return lray;
}

#ifdef WITH_CUDA

struct SunLightData {
  Vector3f orientation;
  Vector3f intensity;
};

KERNEL void UpdatePointLight(Entity *dst_b, SunLightData params) {
  auto *dst = static_cast<SunLight*>(dst_b);
  dst->orientation = params.orientation;
  dst->intensity = params.intensity;
}

void SunLight::UpdateDevice(Entity *device_ptr) const {
  SunLightData params;
  params.orientation = orientation;
  params.intensity = intensity;
  UpdatePointLight<<<1, 1>>>(device_ptr, params);
}


#else

void SunLight::UpdateDevice(Entity*) const {}

#endif

void SunLight::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "SetIntensity") {
    ParseArg(args[1], intensity);
  } else if (args[0] == "SetOrientation") {
    ParseArg(args[1], orientation);
  } else {
    throw UnknownCommand(args[0]);
  }
}

void SunLight::Save(SceneBuilder *build, FnSaveArgs fn_write) const {
  (void)build;
  fn_write({"SetIntensity", ToString(intensity)});
  fn_write({"SetOrientation", ToString(orientation)});
}

CPU_AND_CUDA LightRay PointLight::Li(const Point3f &p) const {
  LightRay lray;
  Vector3f path = (position - p);
  lray.dist = path.Norm();
  lray.wi = path / lray.dist;
  lray.spectrum = intensity / (lray.dist * lray.dist);
  return lray;
}

#ifdef WITH_CUDA

struct PointLightData {
  Vector3f position;
  Vector3f intensity;
};

KERNEL void UpdatePointLight(Entity *dst_b, PointLightData params) {
  auto *dst = static_cast<PointLight*>(dst_b);
  dst->position = params.position;
  dst->intensity = params.intensity;
}

void PointLight::UpdateDevice(Entity *device_ptr) const {
  PointLightData params;
  params.position = position;
  params.intensity = intensity;
  UpdatePointLight<<<1, 1>>>(device_ptr, params);
}


#else

void PointLight::UpdateDevice(Entity*) const {}

#endif

void PointLight::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "SetIntensity") {
    ParseArg(args[1], intensity);
  } else if (args[0] == "SetPosition") {
    ParseArg(args[1], position);
  } else {
    throw UnknownCommand(args[0]);
  }
}

void PointLight::Save(SceneBuilder *build, FnSaveArgs fn_write) const {
  (void)build;
  fn_write({"SetIntensity", ToString(intensity)});
  fn_write({"SetPosition", ToString(position)});
}

}
