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

#ifndef QJULIA_CAMERA3D_H_
#define QJULIA_CAMERA3D_H_

#include "core/camera.h"

namespace qjulia {

class Camera3D : public Camera {
 public:
  
  CPU_AND_CUDA Camera3D(void);
  
  CPU_AND_CUDA void LookAt(Vector3f position, Vector3f at, Vector3f up);
  
  CPU_AND_CUDA void CenterAround(Float h, Float v, Float radius);
  
  // TODO: what to do if parsed from file?
  CPU_AND_CUDA void Update(void);
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  Point3f position;
  Point3f orientation;
  Point3f up;
  Point3f right;
};

/** \brief Standard 3D camera with orthogonal projection
*/
class OrthoCamera : public Camera3D {
 public:
  CPU_AND_CUDA OrthoCamera(void) {}
  
  CPU_AND_CUDA Ray CastRay(Point2f pos) const override;
  
  Float scale = 1;
};

/** \brief Standard 3D camera with perspective projection
*/
class PerspectiveCamera : public Camera3D {
 public:
  CPU_AND_CUDA PerspectiveCamera(void);
  
  CPU_AND_CUDA Ray CastRay(Point2f pos) const override;
  
  void Parse(const Args &args, SceneBuilder *build) override;
  
  Float focus = 1.8;
};

}

#endif
