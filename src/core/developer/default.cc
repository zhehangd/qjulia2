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

#include "core/developer/default.h"
#include "core/algorithm.h"
#include "core/color.h"

namespace qjulia {

namespace {

/*
CPU_AND_CUDA Pixel DevelopPixel(const Sample &sample) {
  Pixel pix;
  for (int k = 0; k < 3; ++k) {
    pix[k] = (unsigned char)round(
      min((Float)255, max((Float)0.0, sample.spectrum[k] * 255)));
  }
  return pix;
}

CPU_AND_CUDA Pixel DevelopPixelDepth(const Sample &sample) {
  Pixel pix(0, 0, 0);
  if (!sample.has_isect) {return pix;}
  Float min_dist = 2.85;
  Float max_dist = 3.85;
  Float dist = (sample.depth - min_dist) / (max_dist - min_dist);
  Vector3f color(dist, dist, dist);
  pix = ClipTo8Bit(color * 255);
  return pix;
}*/

}

void DefaultDeveloper::Develop(const Film &film, float w) {
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &src = film.At(i);
    auto &dst = cache1_.At(i);
    dst.spectrum += src.spectrum * w;
    dst.w += w;
    if (!std::isnan(src.depth)) {
      if (std::isnan(dst.depth)) {
        dst.depth = src.depth * w;
        dst.depth_w = w;
      } else {
        dst.depth += src.depth * w;
        dst.depth_w += w;
      }
    }
  }
}

void DefaultDeveloper::Init(Size size) {
  cache1_.Resize(size);
  cache1_.SetTo({});
}
  
void DefaultDeveloper::Finish(void) {
  //ProduceImage(dst);
}

void DefaultDeveloper::ProduceImage(RGBImage &dst) {
  dst.Resize(cache1_.ArraySize());
  for (int i = 0; i < dst.NumElems(); ++i) {
    auto &src = cache1_.At(i);
    dst.At(i) = ClipTo8Bit(src.spectrum * (255.0 / src.w));
  }
}

void DefaultDeveloper::ProduceImage(RGBFloatImage &dst) {
  dst.Resize(cache1_.ArraySize());
  for (int i = 0; i < dst.NumElems(); ++i) {
    auto &src = cache1_.At(i);
    dst.At(i) = src.spectrum / src.w;
  }
}

void DefaultDeveloper::ProduceDepthImage(GrayscaleFloatImage &dst) {
  dst.Resize(cache1_.ArraySize());
  for (int i = 0; i < dst.NumElems(); ++i) {
    auto &src = cache1_.At(i);
    dst.At(i) = src.depth / src.depth_w;
  }
}

void DefaultDeveloper::RetrieveFromDevice(Developer *device_ptr) {
  (void)device_ptr;
}

}

