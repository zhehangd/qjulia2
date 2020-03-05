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

#include "core/scene.h"

#include "core/base.h"
#include "core/intersection.h"
#include "core/resource_mgr.h"

namespace qjulia {

const Object* Scene::Intersect(const Ray &ray, Intersection *isect) const {
  isect->Clear();
  Float min_dist = std::numeric_limits<Float>::infinity();
  const Object *nearest_object = nullptr;
  Intersection curr_isect;
  for (int i = 0; i < NumObjects(); ++i) {
    const Object *object = GetObject(i);
    assert(object);
    curr_isect = object->Intersect(ray);    
    if (curr_isect.good == false) {continue;}
    if (curr_isect.dist >= min_dist) {continue;}
    nearest_object = object;
    min_dist = curr_isect.dist;
    *isect = curr_isect;
  }
  return nearest_object;
}

bool Scene::ParseInstruction(const TokenizedStatement instruction, 
                             const ResourceMgr *resource) {
  // Empty
  if (instruction.size() == 0) {return true;}
  
  // Object
  if (instruction[0] == GetEntityTypeName(Object::kType)) {
    const Object *p = nullptr;
    bool good = ParseInstruction_Pointer<Object>(
      instruction, resource, &p);
    if (good) {objects_.push_back(p);}
    return good;
  
  // Light
  } else if (instruction[0] == GetEntityTypeName(Light::kType)) {
    const Light *p = nullptr;
    bool good = ParseInstruction_Pointer<Light>(
      instruction, resource, &p);
    if (good) {lights_.push_back(p);}
    return good;
  
  // Camera
  } else if (instruction[0] == GetEntityTypeName(Camera::kType)) {
    const Camera *p = nullptr;
    bool good = ParseInstruction_Pointer<Camera>(
      instruction, resource, &p);
    if (good) {cameras_.push_back(p);}
    return good;
  
  } else {
    return UnknownInstructionError(instruction);
  }
}

}
