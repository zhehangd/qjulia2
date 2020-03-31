#ifndef QJULIA_H_
#define QJULIA_H_

#include <string>
#include <memory>

#include "array2d.h"
#include "vector.h"

namespace qjulia {

class Film;

struct Image : public Array2D<Pixel> {
  Image(void) {}
  Image(int w, int h) : Array2D<Pixel>({w, h}) {}
  Image(const Film &film);
  
  int BytesPerRow(void) const {return 3 * Width();}
};

void UpSample(const Image &src, Image &dst, Size size);

void ConvertFilmToImage(const Film &film, Image &image);

void ReadPngImage(std::string filename); 

void WritePngImage(std::string filename, Image &image);

}

#endif
