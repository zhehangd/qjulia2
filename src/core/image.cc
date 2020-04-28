#include "image.h"

#define PNG_DEBUG 3

#include <stdlib.h>

#include <png.h>

#include <glog/logging.h>

#include "film.h"

namespace qjulia {

void UpSample(const Image &src, Image &dst, Size size) {
  dst.Resize(size);
  for (int dr = 0; dr < dst.Height(); ++dr) {
    for (int dc = 0; dc < dst.Width(); ++dc) {
      int sr = dr * src.Height() / dst.Height();
      int sc = dc * src.Width() / dst.Width();
      dst.At(dr, dc) = src.At(sr, sc);
    }
  }
}

}
