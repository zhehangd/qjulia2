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

#include "object.h"

#include "base.h"
#include "shape.h"
#include "transform.h"
#include "material.h"
#include "resource_mgr.h"

namespace qjulia {

Intersection Object::Intersect(const Ray &ray) const {
  Intersection isect;
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

bool Object::ParseInstruction(const TokenizedStatement instruction, 
                             const ResourceMgr *resource) {
  // Empty
  if (instruction.size() == 0) {return true;}
  
  // Shape
  if (instruction[0] == GetEntityTypeName(Shape::kType)) {
    return ParseInstruction_Pointer<Shape>(
      instruction, resource, &shape);
  
  // Material
  } else if (instruction[0] == GetEntityTypeName(Material::kType)) {
    return ParseInstruction_Pointer<Material>(
      instruction, resource, &material);
  
  // Transform
  } else if (instruction[0] == GetEntityTypeName(Transform::kType)) {
    return ParseInstruction_Pointer<Transform>(
      instruction, resource, &transform);
  
  } else {
    return UnknownInstructionError(instruction);
  }
}

}
