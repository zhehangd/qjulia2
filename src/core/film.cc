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

#include "core/film.h"

#include <math.h>

#include <fstream>
#include <sstream>

namespace qjulia {

CPU_AND_CUDA Point2f GenerateCameraCoords(Point2f src, Size size) {
  Float c = src[1];
  Float r=  src[0];
  Float ss = (Float)(size.width < size.height ? size.width : size.height - 1);
  Float x = (c - (size.width - 1) * 0.5f) / ss;
  Float y = ((size.height - 1) * 0.5f - r) / ss;
  return {x, y};
}

/*
void SaveToPPM(const std::string &filename, const Film &film, Float scale) {
  int w = film.Width();
  int h = film.Height();
  std::vector<unsigned char> buf(w * h * 3);
  auto *p = buf.data();
  for (int i = 0; i < (w * h); ++i) {
    const auto &sp = film.At(i);
    for (int ch = 0; ch < 3; ++ch) {
      *(p++) = std::min(255, std::max(0, (int)std::round(sp[ch] * scale)));
    }
  }
  std::ostringstream header_stream;
  header_stream << "P6 " << w << ' ' << h << ' ' << 255 << '\n';
  std::string header = header_stream.str();
  std::ofstream file_stream(filename, std::ofstream::binary);
  file_stream.write(header.c_str(), header.size());
  file_stream.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}*/

}
