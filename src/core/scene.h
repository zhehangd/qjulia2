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

namespace qjulia {

class Object;
class Light;

class Scene : public SceneEntity {
 public:
   
  EntityType GetType(void) const final {return kType;}
  
  void AddObject(Object *obj) {objects_.emplace_back(obj);}
  void AddLight(Light *light) {lights_.emplace_back(light);}
  void AddCamera(Camera *camera) {cameras_.emplace_back(camera);}
  
  // TODO
  void SetActiveCamera(int i = 0) {active_camera_ = cameras_[i];}
  const Camera* GetActiveCamera(void) const {return active_camera_;}
  
  const Object* Intersect(const Ray &ray, Intersection *isect) const;
  
  int NumObjects(void) const {return objects_.size();}
  int NumLights(void) const {return lights_.size();}
  
  const Object* GetObject(int i) const {return objects_[i];}
  
  const Light* GetLight(int i) const {return lights_[i];}
  
  bool AddShape(const std::string &name, Shape* shape);
  
  SceneEntity* Clone(void) const override {return new Scene(*this);}
  
  bool ParseInstruction(const TokenizedStatement instruction, 
                        const ResourceMgr *resource) override;
  
  static const EntityType kType = EntityType::kScene;
  
  std::vector<const Object*> objects_;
  std::vector<const Light*> lights_;
  std::vector<const Camera*> cameras_;
  const Camera *active_camera_;
};

}

#endif
