#ifndef QJULIA_H_
#define QJULIA_H_

#include <string>
#include <memory>

#include "array2d.h"
#include "vector.h"

namespace qjulia {
  
struct Image : public Array2D<Pixel> {
  Image(int w, int h) : Array2D<Pixel>({w, h}) {}
};

void ReadPngImage(std::string filename); 

void WritePngImage(std::string filename, Image &image);

}

#endif
