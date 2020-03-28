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

#ifndef QJULIA_SCENE_H_
#define QJULIA_SCENE_H_

#include <vector>
#include <memory>

#include "entity.h"
#include "camera.h"
#include "intersection.h"
#include "light.h"
#include "material.h"
#include "object.h"
#include "shape.h"
#include "transform.h"
#include "vector.h"
#include "world.h"

namespace qjulia {

class Scene {
 public:
  
  CPU_AND_CUDA const Camera* GetCamera(void) const {return camera_;}
  
  CPU_AND_CUDA const Object* Intersect(const Ray &ray, Intersection *isect) const {return world_->Intersect(ray, isect);}
  
  CPU_AND_CUDA int NumObjects(void) const {return world_->NumObjects();}
  CPU_AND_CUDA int NumLights(void) const {return world_->NumLights();}
  
  CPU_AND_CUDA const Object* GetObject(int i) const {return world_->GetObject(i);}
  
  CPU_AND_CUDA const Light* GetLight(int i) const {return world_->GetLight(i);}
  
  World *world_;
  Camera *camera_;
};

}

#endif
