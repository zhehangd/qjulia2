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

#include "core/camera/camera3d.h"

#include "core/arg_parse.h"

namespace qjulia {

struct Camera3DData {
  Point3f position;
  Point3f orientation;
  Point3f up;
  Point3f right;
};

struct PerspectiveCameraData : public Camera3DData {
  Float focus;
};

struct OrthoCameraData : public Camera3DData {
  Float scale;
};

CPU_AND_CUDA Ray OrthoCamera::CastRay(Point2f pos) const {
  Float x = pos[0];
  Float y = pos[1];
  return Ray(position + right * x + up * y, orientation);
}

#ifdef WITH_CUDA

KERNEL void UpdateOrthoCamera(Entity *dst_b, OrthoCameraData params) {
  auto *dst = static_cast<OrthoCamera*>(dst_b);
  dst->position = params.position;
  dst->orientation = params.orientation;
  dst->up = params.up;
  dst->right = params.right;
  dst->scale = params.scale;
}

void OrthoCamera::UpdateDevice(Entity *device_ptr) const {
  OrthoCameraData params;
  params.position = position;
  params.orientation = orientation;
  params.up = up;
  params.right = right;
  params.scale = scale;
  UpdateOrthoCamera<<<1, 1>>>(device_ptr, params);
}

#else

void OrthoCamera::UpdateDevice(Entity*) const {}

#endif

void OrthoCamera::Parse(const Args &args, SceneBuilder *build) {
  CHECK(args.size() > 0);
  if (args[0] == "SetScale") {
    ParseArg(args[1], scale);
  } else {
    Camera::Parse(args, build);
  }
}

CPU_AND_CUDA PerspectiveCamera::PerspectiveCamera(void) {
  position = {0, 0, 1};
  orientation = {0, 0, -1};
  up = {0, 1, 0};
  focus = 1;
}

CPU_AND_CUDA Ray PerspectiveCamera::CastRay(Point2f pos) const {
  Float x = pos[0];
  Float y = pos[1];
  Float z = focus;
  Point3f dir = Normalize(orientation * z + up * y + right * x);
  return Ray(position, dir);
}

#ifdef WITH_CUDA

KERNEL void UpdatePerspectiveCamera(Entity *dst_b, PerspectiveCameraData params) {
  auto *dst = static_cast<PerspectiveCamera*>(dst_b);
  dst->position = params.position;
  dst->orientation = params.orientation;
  dst->up = params.up;
  dst->right = params.right;
  dst->focus = params.focus;
}

void PerspectiveCamera::UpdateDevice(Entity *device_ptr) const {
  PerspectiveCameraData params;
  params.position = position;
  params.orientation = orientation;
  params.up = up;
  params.right = right;
  params.focus = focus;
  UpdatePerspectiveCamera<<<1, 1>>>(device_ptr, params);
}

#else

void PerspectiveCamera::UpdateDevice(Entity*) const {}

#endif

void PerspectiveCamera::Parse(const Args &args, SceneBuilder *build) {
  CHECK(args.size() > 0);
  if (args[0] == "SetFocus") {
    ParseArg(args[1], focus);
  } else {
    Camera::Parse(args, build);
  }
}

}
