#include "image_io.h"

#include <algorithm>
#include <cctype>

#include "core/image_io/image_io_tiff.h"
#include "core/image_io/image_io_png.h"

namespace qjulia {

ImageFormat DetectImageFormat(std::string filename) {
  auto dot_pos = filename.find_last_of(".");
  if (dot_pos == std::string::npos) {return ImageFormat::kOther;}
  auto ext = filename.substr(dot_pos + 1);
  std::for_each(ext.begin(), ext.end(), [](char &c){c = std::toupper(c);});
  if (ext == "TIFF" || ext == "TIF") {return ImageFormat::kTIFF;}
  if (ext == "PNG") {return ImageFormat::kPNG;}
  return ImageFormat::kOther;
}

template <typename ImageType>
void ImwriteTemplate(std::string filename, const ImageType &image) {
  ImageFormat format_code = DetectImageFormat(filename);
  CHECK(format_code != ImageFormat::kOther);
  if (format_code == ImageFormat::kPNG) {
    PNGImageWriter writer;
    writer.WriteImage(filename, image);
  } else if (format_code == ImageFormat::kTIFF) {
    TIFFImageWriter writer;
    writer.WriteImage(filename, image);
  }
}

void Imwrite(std::string filename, const RGBImage &image) {
  ImwriteTemplate(filename, image);
}

void Imwrite(std::string filename, const RGBFloatImage &image) {
  ImwriteTemplate(filename, image);
}

void Imwrite(std::string filename, const GrayscaleImage &image) {
  ImwriteTemplate(filename, image);
}

void Imwrite(std::string filename, const GrayscaleFloatImage &image) {
  ImwriteTemplate(filename, image);
}

template <typename ImageType>
void ImreadTemplate(std::string filename, ImageType &image) {
  ImageFormat format_code = DetectImageFormat(filename);
  CHECK(format_code != ImageFormat::kOther);
  if (format_code == ImageFormat::kPNG) {
    PNGImageReader reader;
    reader.ReadImage(filename, image);
  } else if (format_code == ImageFormat::kTIFF) {
    TIFFImageReader reader;
    reader.ReadImage(filename, image);
  }
}

void Imread(std::string filename, RGBImage &image) {
  ImreadTemplate(filename, image);
}

void Imread(std::string filename, RGBFloatImage &image) {
  ImreadTemplate(filename, image);
}

void Imread(std::string filename, GrayscaleImage &image) {
  ImreadTemplate(filename, image);
}

void Imread(std::string filename, GrayscaleFloatImage &image) {
  ImreadTemplate(filename, image);
}

}
