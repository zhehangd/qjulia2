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

#ifndef QJULIA_WORLD_H_
#define QJULIA_WORLD_H_

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

namespace qjulia {

/// @brief Scene
///
/// 
class World : public Entity {
 public:
  
  CPU_AND_CUDA World(void); 
  
  CPU_AND_CUDA const Object* Intersect(const Ray &ray, Intersection *isect) const;
  
  CPU_AND_CUDA int NumObjects(void) const {return data_.num_objects;}
  CPU_AND_CUDA int NumLights(void) const {return data_.num_lights;}
  
  CPU_AND_CUDA const Object* GetObject(int i) const {return data_.objects[i];}
  
  CPU_AND_CUDA const Light* GetLight(int i) const {return data_.lights[i];}
  
  void UpdateDevice(Entity *device_ptr) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  void Save(SceneBuilder *build, FnSaveArgs fn_write) const override;
  
  struct Data {
    int num_objects = 0;
    int num_lights = 0;
    int num_cameras = 0;
    Object* objects[20] = {};
    Light* lights[20] = {};
    Camera* cameras[20] = {};
  };
  
  Data data_device_;
  Data data_host_;
  Data &data_;
};

}

#endif
 
