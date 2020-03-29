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

#include "core/transform.h"

#include "core/arg_parse.h"
#include "core/scene_descr.h"

namespace qjulia {

namespace {
}

std::ostream& operator<<(std::ostream &os, const Matrix4x4 &mat) {
  os << '\n';
  for (int i = 0; i < 4; ++i) {
    os << mat.At(i, 0) << ", " << mat.At(i, 1) << ", "
       << mat.At(i, 2) << ", " << mat.At(i, 3) << "\n";
  }
  return os;
}

void Transform::Parse(const Args &args, SceneBuilder *build) {
  (void)build;
  if (args.size() == 0) {return;}
  if (args[0] == "SetTranslate") {
    Vector3f t;
    ParseArg(args[1], t);
    mat_ow_ = Matrix4x4::Translate(t) * mat_ow_;
    mat_wo_ = mat_wo_ * Matrix4x4::Translate(-t);
  } else if (args[0] == "SetScale") {
    Float s;
    ParseArg(args[1], s);
    mat_ow_ = Matrix4x4::Scale({s, s, s}) * mat_ow_;
    mat_wo_ = mat_wo_ * Matrix4x4::Scale({1/s, 1/s, 1/s});
  } else if (args[0] == "SetRotate") {
    std::string axis = args[1];
    float angle;
    ParseArg(args[2], angle);
    Matrix4x4 (*rot)(const float) = nullptr;
    if (axis == "x") {
      rot = Matrix4x4::RotateX;
    } else if (axis == "y") {
      rot = Matrix4x4::RotateY;
    } else if (axis == "z") {
      rot = Matrix4x4::RotateZ;
    } else {
      LOG(FATAL) << "Error: Unkown axis " << axis << ".";
    }
    mat_ow_ = (*rot)(angle) * mat_ow_;
    mat_wo_ = mat_wo_ * (*rot)(-angle);
  } else {
    throw UnknownCommand(args[0]);
  }
}

}
