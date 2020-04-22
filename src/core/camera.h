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

#ifndef QJULIA_CAMERA_H_
#define QJULIA_CAMERA_H_

#include "entity.h"
#include "ray.h"
#include "vector.h"

namespace qjulia {

/// @brief Defines how rays are projected to films and vice versa
///
/// This is the interface that all camera classes should implement
///
class Camera : public Entity {
 public:
  
  /// @brief Cast a ray from the camera to the scene
  CPU_AND_CUDA virtual Ray CastRay(Point2f pos) const = 0;
  
  CPU_AND_CUDA Camera(void);
  
  CPU_AND_CUDA void LookAt(Vector3f position, Vector3f at, Vector3f up);
  
  CPU_AND_CUDA void CenterAround(Float h, Float v, Float radius);
  
  CPU_AND_CUDA Point3f GetPosition(void) const {return position;}
  
  CPU_AND_CUDA Point3f GetTarget(void) const {return target_;}
  
  // TODO: what to do if parsed from file?
  CPU_AND_CUDA void Update(void);
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  Point3f target_;
  Point3f position;
  Point3f up;
  
  Point3f orientation;
  Point3f right;
};

}

#endif
