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
#include "core/color.h"

namespace qjulia {

namespace {
}


CPU_AND_CUDA unsigned char ClipTo8Bit(Float v) {
  return (unsigned char)round(min((Float)255, max((Float)0.0, v)));
}

CPU_AND_CUDA Pixel ClipTo8Bit(Vector3f v) {
  return Pixel(ClipTo8Bit(v[0]), ClipTo8Bit(v[1]), ClipTo8Bit(v[2]));
}

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
}

CPU_AND_CUDA void DefaultDeveloper::Develop(const Film &film, float w) {
  for (int i = 0; i < film.NumElems(); ++i) {
    cache1_.At(i).spectrum += film.At(i).spectrum;
    cache1_.At(i).w += w;
  }
}

CPU_AND_CUDA void DefaultDeveloper::Init(Size size) {
  cache1_.Resize(size);
  cache1_.SetTo({});
}
  
CPU_AND_CUDA void DefaultDeveloper::Finish(Image &dst) {
  dst.Resize(cache1_.ArraySize());
  for (int i = 0; i < dst.NumElems(); ++i) {
    auto &src = cache1_.At(i);
    dst.At(i) = ClipTo8Bit(src.spectrum * (255.0 / src.w));
  }
}

}

