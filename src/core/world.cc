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

#include "core/world.h"

#include "core/base.h"
#include "core/intersection.h"
#include "core/object.h"
#include "core/scene_builder.h"
#include "core/scene_descr.h"

namespace qjulia {

CPU_AND_CUDA const Object* World::Intersect(const Ray &ray, Intersection *isect) const {
  isect->Clear();
  Float min_dist = 99999999;//std::numeric_limits<Float>::infinity();
  const Object *nearest_object = nullptr;
  Intersection curr_isect;
  for (int i = 0; i < NumObjects(); ++i) {
    const Object *object = GetObject(i);
    curr_isect = object->Intersect(ray);    
    if (curr_isect.good == false) {continue;}
    if (curr_isect.dist >= min_dist) {continue;}
    nearest_object = object;
    min_dist = curr_isect.dist;
    *isect = curr_isect;
  }
  return nearest_object;
}

CPU_AND_CUDA World::World(void)
#ifdef __CUDA_ARCH__
  : data_(data_device_) {
#else
  : data_(data_host_) {
#endif
}

void World::Parse(const Args &args, SceneBuilder *build) {
  if (args.size() == 0) {return;}
  if (args[0] == "AddObject") {
    auto *node = ParseEntityNode<Object>(args[1], build);
    data_host_.objects[data_host_.num_objects++] = CHECK_NOTNULL(node->Get());
    data_device_.objects[data_device_.num_objects++] = node->GetDevice();
  } else if (args[0] == "AddCamera") {
    auto *node = ParseEntityNode<Camera>(args[1], build);
    data_host_.cameras[data_host_.num_cameras++] = CHECK_NOTNULL(node->Get());
    data_device_.cameras[data_device_.num_cameras++] = node->GetDevice();
  } else if (args[0] == "AddLight") {
    auto *node = ParseEntityNode<Light>(args[1], build);
    data_host_.lights[data_host_.num_lights++] = CHECK_NOTNULL(node->Get());
    data_device_.lights[data_device_.num_lights++] = node->GetDevice();
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
