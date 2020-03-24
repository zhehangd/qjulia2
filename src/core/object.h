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

#ifndef QJULIA_OBJECT_H_
#define QJULIA_OBJECT_H_

#include <vector>
#include <memory>

#include "array2d.h"
#include "entity.h"
#include "intersection.h"
#include "ray.h"
#include "spectrum.h"
#include "vector.h"

namespace qjulia {

class Shape;
class Transform;
class Material;

class Object : public SceneEntity {
 public:
  
  EntityType GetType(void) const final {return kType;}
  
  /** \brief Test ray itersection with transformed shape.
  */
  Intersection Intersect(const Ray &ray) const;
  
  SceneEntity* Clone(void) const {return new Object(*this);}
  
  bool ParseInstruction(const TokenizedStatement instruction, 
                        const ResourceMgr *resource) override;
  
  static const EntityType kType = EntityType::kObject;
  
  const Shape *shape = nullptr;
  const Material *material = nullptr;
  const Transform *transform = nullptr;
};

}

#endif
 
