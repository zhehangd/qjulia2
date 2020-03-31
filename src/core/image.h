#ifndef QJULIA_H_
#define QJULIA_H_

#include <string>
#include <memory>

struct Image {
  unsigned char *data;
  int w;
  int h;
};

Image ReadPngImage(std::string filename); 

void WritePngImage(std::string filename, Image image);

#endif
