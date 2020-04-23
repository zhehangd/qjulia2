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

CPU_AND_CUDA Pixel DevelopPixel(const Sample &sample) {
  Pixel pix;
  for (int k = 0; k < 3; ++k) {
    pix[k] = (unsigned char)round(
      min((Float)255, max((Float)0.0, sample.spectrum[k] * 255)));
  }
  return pix;
}

CPU_AND_CUDA Pixel DevelopPixel2(const Sample &sample) {
  Pixel pix(0, 0, 0);
  if (!sample.has_isect) {return pix;}
  Float lumin = (sample.spectrum[0] + sample.spectrum[1] + sample.spectrum[2]) / 3.0;
  Float dist = sample.depth;
  Float min_dist = 1;
  Float max_dist = 6;
  Float hue = (dist - min_dist) / (max_dist - min_dist) * 360;
  hue = max(min(hue, (Float)360.0), (Float)0);
  auto color = LCH2RGB({lumin, 0.4, hue});
  for (int k = 0; k < 3; ++k) {pix[k] = ClipTo8Bit(color[k] * 255);}
  return pix;
}

CPU_AND_CUDA void Blur(const Film &src, Film &dst) {
  dst.Resize(src.ArraySize());
  int w = src.Width();
  int h = src.Height();
  for (int r = 5; r < h - 5; ++r) {
    for (int c = 5; c < w - 5; ++c) {
      dst.At(r, c) = src.At(r, c);
    }
  }
}

CPU_AND_CUDA void DefaultDeveloper::Develop(const Film &film, Image &image) {
  Film cache(film.ArraySize());
  image.Resize(film.ArraySize());
  for (int i = 0; i < film.NumElems(); ++i) {
    auto &src = film.At(i);
    auto &dst = image.At(i);
    dst = DevelopPixel(src);
  }
}

}

