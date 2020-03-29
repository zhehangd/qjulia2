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

#include "core/object.h"

#include "core/arg_parse.h"
#include "core/base.h"
#include "core/camera.h"
#include "core/light.h"
#include "core/shape.h"
#include "core/transform.h"
#include "core/material.h"
#include "core/scene_descr.h"
#include "core/scene_builder.h"

namespace qjulia {

CPU_AND_CUDA Object::Object(void)
#ifdef __CUDA_ARCH__
  : data_(data_device_) {
#else
  : data_(data_host_) {
#endif
}

CPU_AND_CUDA Intersection Object::Intersect(const Ray &ray) const {
  Intersection isect;
  auto *transform = data_.transform;
  auto *shape = data_.shape;
  if (shape == nullptr) {return isect;}
  if (transform != nullptr) {
    Ray ray_local = ray;
    ray_local.start = transform->W2O_Point(ray_local.start);
    ray_local.dir = Normalize(transform->W2O_Vector(ray_local.dir));
    isect = shape->Intersect(ray_local);
    isect.position = transform->O2W_Point(isect.position);
    isect.normal = transform->O2W_Normal(isect.normal);
    isect.dist = transform->O2W_Vector(ray_local.dir * isect.dist).Norm();
  } else {
    isect = shape->Intersect(ray);
  }
  return isect;
}

#ifdef WITH_CUDA

KERNEL void UpdateObject(Entity *dst_b, Object::Data data_host,
                        Object::Data data_device) {
  auto *dst = static_cast<Object*>(dst_b);
  dst->data_host_ = data_host;
  dst->data_device_ = data_device;
}

void Object::UpdateDevice(Entity *device_ptr) const {
  UpdateObject<<<1, 1>>>(device_ptr, data_host_, data_device_);
}

#endif

void Object::Parse(const Args &args, SceneBuilder *build) {
  if (args.size() == 0) {return;}
  if (args[0] == "SetShape") {
    auto *node = ParseEntityNode<Shape>(args[1], build);
    data_host_.shape = node->Get();
    data_device_.shape = node->GetDevice();
  } else if (args[0] == "SetMaterial") {
    auto *node = ParseEntityNode<Material>(args[1], build);
    data_host_.material = node->Get();
    data_device_.material = node->GetDevice();
  } else if (args[0] == "SetTransform") {
    auto *node = ParseEntityNode<Transform>(args[1], build);
    data_host_.transform = node->Get();
    data_device_.transform = node->GetDevice();
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
