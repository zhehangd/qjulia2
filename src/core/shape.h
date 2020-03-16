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

#ifndef QJULIA_SHAPE_H_
#define QJULIA_SHAPE_H_

#include <vector>
#include <memory>

#include "array2d.h"
#include "entity.h"
#include "intersection.h"
#include "vector.h"
#include "ray.h"

namespace qjulia {

class Scene;

class Shape : public SceneEntity {
 public:
  
  EntityType GetType(void) const final {return kType;}
  
  virtual Intersection Intersect(const Ray &ray) const = 0;
  
  virtual void Intersect(const Array2D<Ray> &rays,
                         Array2D<Intersection> &isects) const {
    for (int i = 0; i < rays.NumElems(); ++i) {
      isects(i) = Intersect(rays(i));
    }
  }
  
  SceneEntity* Clone(void) const override = 0;
  
  bool ParseInstruction(const TokenizedStatement instruction, 
                        const ResourceMgr *resource) override = 0;
  
  static const EntityType kType = EntityType::kShape;
};

}

#endif
 
 
