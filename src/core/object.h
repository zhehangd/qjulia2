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
#include "texture.h"
#include "vector.h"

namespace qjulia {

class Shape;
class Transform;
class Material;

class Object : public Entity {
 public:
  
  CPU_AND_CUDA Object(void); 
  
  /** \brief Test ray itersection with transformed shape.
  */
  CPU_AND_CUDA Intersection Intersect(const Ray &ray) const;
  
  
  CPU_AND_CUDA const Material* GetMaterial(void) const {return data_.material;}
  
  CPU_AND_CUDA const Shape* GetShape(void) const {return data_.shape;}
  
  CPU_AND_CUDA const Transform* GetTransform(void) const {return data_.transform;}
  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  struct Data {
    const Shape *shape = nullptr;
    const Material *material = nullptr;
    const Transform *transform = nullptr;
  };
  
  Data data_device_;
  Data data_host_;
  Data &data_;
};

}

#endif
 
