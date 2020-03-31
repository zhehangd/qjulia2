#ifndef QJULIA_H_
#define QJULIA_H_

#include <string>
#include <memory>

#include "array2d.h"
#include "vector.h"

namespace qjulia {

class Film;

struct Image : public Array2D<Pixel> {
  Image(int w, int h) : Array2D<Pixel>({w, h}) {}
  Image(const Film &film);
};

void ReadPngImage(std::string filename); 

void WritePngImage(std::string filename, Image &image);

}

#endif
