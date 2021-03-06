#ifndef QJULIA_IMAGE_IO_PNG_H_
#define QJULIA_IMAGE_IO_PNG_H_

#include "core/image_io.h"

namespace qjulia {

class PNGImageReader : public ImageReaderInterface {
 public:
  void ReadImage(const std::string &filename, RGBImage &image) override;
  void ReadImage(const std::string &filename, RGBFloatImage &image) override;
  void ReadImage(const std::string &filename, GrayscaleImage &image) override;
  void ReadImage(const std::string &filename, GrayscaleFloatImage &image) override;
};

class PNGImageWriter : public ImageWriterInterface {
 public:
  
  void WriteImage(const std::string &filename, const RGBImage &image) override;
  
  void WriteImage(const std::string &filename, const RGBFloatImage &image) override;
  
  void WriteImage(const std::string &filename, const GrayscaleImage &image) override;
  
  void WriteImage(const std::string &filename, const GrayscaleFloatImage &image) override;
  
};

}

#endif
