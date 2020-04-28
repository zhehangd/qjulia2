#ifndef QJULIA_H_
#define QJULIA_H_

#include <string>
#include <memory>

#include "array2d.h"
#include "vector.h"

namespace qjulia {

typedef Array2D<Vector3b> RGBImage;
typedef Array2D<Vector3f> RGBFloatImage;
typedef Array2D<Byte> GrayscaleImage;
typedef Array2D<Float> GrayscaleFloatImage;
typedef RGBImage Image;

template <typename ImageType>
void UpSample(const ImageType &src, ImageType &dst, Size size) {
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

#endif
