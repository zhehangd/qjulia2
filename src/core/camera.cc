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

#include "core/camera.h"

#include "core/arg_parse.h"

namespace qjulia {

CPU_AND_CUDA Camera::Camera(void) {
  position = {0, 0, 1};
  orientation = {0, 0, -1};
  up = {0, 1, 0};
}

CPU_AND_CUDA void Camera::Update(void) {
  orientation = Normalize(orientation);
  up = up - Project(up, orientation);
  up = Normalize(up);
  right = Cross(orientation, up);
}

CPU_AND_CUDA void Camera::LookAt(Vector3f position, Vector3f at, Vector3f up) {
  target_ = at;
  this->position = position;
  orientation = at - position;
  this->up = up;
  Update();
}

CPU_AND_CUDA void Camera::CenterAround(Float h, Float v, Float radius) {
  h *= kPi / 180.0f;
  v *= kPi / 180.0f;
  Float x = std::sin(h) * std::cos(v) * radius;
  Float z = std::cos(h) * std::cos(v) * radius;
  Float y = std::sin(v) * std::sin(v) * radius;
  LookAt({x, y, z}, {0, 0, 0}, {0, 1, 0});
}

void Camera::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  
  CHECK(args.size() > 0);
  if (args[0] == "LookAt") {
    Vector3f v3[3];
    bool good = true;
    ParseArg(args[1], v3[0]);
    ParseArg(args[2], v3[1]);
    ParseArg(args[3], v3[2]);
    if (good) {
      LookAt(v3[0], v3[1], v3[2]);
    } else {
      LOG(FATAL) << "Not GOOD instruction";
    }
  } else if (args[0] == "SetPosition") {
    ParseArg(args[1], position);
  } else {
    throw UnknownCommand(args[0]);
  }
} 

void Camera::Save(SceneBuilder *build, FnSaveArgs fn_write) const {
  (void)build;
  fn_write({"LookAt", ToString(position), ToString(target_), ToString(up)});
}

}
