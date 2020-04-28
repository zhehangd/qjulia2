#ifndef QJULIA_IMAGE_IO_H_
#define QJULIA_IMAGE_IO_H_

#include <stdexcept>

#include "image.h"

namespace qjulia {

class NoImageSpecificationSupport : public std::runtime_error {
 public:
   NoImageSpecificationSupport(std::string msg) : std::runtime_error(msg.c_str()) {}
};

class ImageReaderInterface {
 public:
  
  virtual ~ImageReaderInterface(void) {}
  
  virtual void ReadImage(const std::string &filename, RGBImage &image) = 0;
  virtual void ReadImage(const std::string &filename, RGBFloatImage &image) = 0;
  virtual void ReadImage(const std::string &filename, GrayscaleImage &image) = 0;
  virtual void ReadImage(const std::string &filename, GrayscaleFloatImage &image) = 0;
};

class ImageWriterInterface {
 public:
  
  virtual ~ImageWriterInterface(void) {}
  
  virtual void WriteImage(const std::string &filename, const RGBImage &image) = 0;
  virtual void WriteImage(const std::string &filename, const RGBFloatImage &image) = 0;
  virtual void WriteImage(const std::string &filename, const GrayscaleImage &image) = 0;
  virtual void WriteImage(const std::string &filename, const GrayscaleFloatImage &image) = 0;
  
};

enum class ImageFormat {
  kPNG,
  kTIFF,
  kOther,
};

ImageFormat DetectImageFormat(std::string filename);

void Imwrite(std::string filename, const RGBImage &image);
void Imwrite(std::string filename, const RGBFloatImage &image);
void Imwrite(std::string filename, const GrayscaleImage &image);
void Imwrite(std::string filename, const GrayscaleFloatImage &image);

void Imread(std::string filename, RGBImage &image);
void Imread(std::string filename, RGBFloatImage &image);
void Imread(std::string filename, GrayscaleImage &image);
void Imread(std::string filename, GrayscaleFloatImage &image);

}

#endif
