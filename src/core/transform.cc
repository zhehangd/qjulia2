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

#include "core/algorithm.h"
#include "core/arg_parse.h"

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

std::array<Vector3f, 3> DecomposeMatrix(const Matrix4x4 &m) {
  // https://nghiaho.com/?page_id=846
  std::array<Vector3f, 3> ret;
  ret[0] = Vector3f(m.At(0, 3), m.At(1, 3), m.At(2, 3));
  ret[2] = Vector3f(
    std::sqrt(m.At(0, 0) * m.At(0, 0)
            + m.At(1, 0) * m.At(1, 0)
            + m.At(2, 0) * m.At(2, 0)),
    std::sqrt(m.At(0, 1) * m.At(0, 1)
            + m.At(1, 1) * m.At(1, 1)
            + m.At(2, 1) * m.At(2, 1)),
    std::sqrt(m.At(0, 2) * m.At(0, 2)
            + m.At(1, 2) * m.At(1, 2)
            + m.At(2, 2) * m.At(2, 2)));
  Float m11 = m.At(0, 0) / ret[2][0];
  Float m21 = m.At(1, 0) / ret[2][0];
  Float m31 = m.At(2, 0) / ret[2][0];
  Float m32 = m.At(2, 1) / ret[2][1];
  Float m33 = m.At(2, 2) / ret[2][2];
  ret[1][0] = Rad2Deg(std::atan2(m32, m33));
  ret[1][1] = Rad2Deg(std::atan2(-m31, std::sqrt(m32 * m32 + m33 * m33)));
  ret[1][2] = Rad2Deg(std::atan2(m21, m11));
  return ret;
}

#ifdef WITH_CUDA

struct TransformData {
  Matrix4x4 mat_ow;
  Matrix4x4 mat_wo;
};

KERNEL void UpdateTransform(Entity *dst_b, TransformData params) {
  auto *dst = static_cast<Transform*>(dst_b);
  dst->mat_ow_ = params.mat_ow;
  dst->mat_wo_ = params.mat_wo;
}

void Transform::UpdateDevice(Entity *device_ptr) const {
  TransformData params;
  params.mat_ow = mat_ow_;
  params.mat_wo = mat_wo_;
  UpdateTransform<<<1, 1>>>(device_ptr, params);
}

#else

void Transform::UpdateDevice(Entity*) const {}

#endif

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
    Float angle;
    ParseArg(args[2], angle);
    Matrix4x4 (*rot)(const Float) = nullptr;
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
  } else if (args[0] == "SetMatrix") {
    ParseArg(args[1], mat_ow_);
  } else {
    throw UnknownCommand(args[0]);
  }
}

void Transform::Save(SceneBuilder *build, FnSaveArgs fn_write) const {
  auto decomp = DecomposeMatrix(mat_ow_);
  fn_write({"SetTranslate", ToString(decomp[0])});
  fn_write({"SetRotate", "x", ToString(decomp[1][0])});
  fn_write({"SetRotate", "y", ToString(decomp[1][1])});
  fn_write({"SetRotate", "z", ToString(decomp[1][2])});
  fn_write({"SetScale", ToString(decomp[2])});
}

}
