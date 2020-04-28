#include "image_io_tiff.h"

#include <cstring>
#include <vector>

#include <tiffio.h>

namespace qjulia {
  
void TIFFImageReader::ReadImage(const std::string &filename, RGBImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "r");
  std::uint32_t w, h;
  std::uint16_t bitdepth;
  std::uint16_t format;
  std::uint16_t compression;
  std::uint16_t channels;
  TIFFGetField(ctx, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(ctx, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(ctx, TIFFTAG_BITSPERSAMPLE, &bitdepth);
  TIFFGetField(ctx, TIFFTAG_SAMPLEFORMAT, &format);
  TIFFGetField(ctx, TIFFTAG_COMPRESSION, &compression);
  TIFFGetField(ctx, TIFFTAG_SAMPLESPERPIXEL, &channels);
  CHECK_EQ(channels, 3);
  CHECK_EQ(bitdepth, 8);
  CHECK_EQ(format, SAMPLEFORMAT_UINT);
  CHECK_EQ(compression, COMPRESSION_NONE);
  image.Resize(Size(w, h));
  for (int r = 0; r < image.Height(); ++r) {
    TIFFReadScanline(ctx, image.Row(r), r);
  }
}

void TIFFImageReader::ReadImage(const std::string &filename, RGBFloatImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "r");
  std::uint32_t w, h;
  std::uint16_t bitdepth;
  std::uint16_t format;
  std::uint16_t compression;
  std::uint16_t channels;
  TIFFGetField(ctx, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(ctx, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(ctx, TIFFTAG_BITSPERSAMPLE, &bitdepth);
  TIFFGetField(ctx, TIFFTAG_SAMPLEFORMAT, &format);
  TIFFGetField(ctx, TIFFTAG_COMPRESSION, &compression);
  TIFFGetField(ctx, TIFFTAG_SAMPLESPERPIXEL, &channels);
  CHECK_EQ(channels, 3);
  CHECK_EQ(bitdepth, 32);
  CHECK_EQ(format, SAMPLEFORMAT_IEEEFP);
  CHECK_EQ(compression, COMPRESSION_NONE);
  image.Resize(Size(w, h));
  for (int r = 0; r < image.Height(); ++r) {
    TIFFReadScanline(ctx, image.Row(r), r);
  }
}

void TIFFImageReader::ReadImage(const std::string &filename, GrayscaleImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "r");
  std::uint32_t w, h;
  std::uint16_t bitdepth;
  std::uint16_t format;
  std::uint16_t compression;
  std::uint16_t channels;
  TIFFGetField(ctx, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(ctx, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(ctx, TIFFTAG_BITSPERSAMPLE, &bitdepth);
  TIFFGetField(ctx, TIFFTAG_SAMPLEFORMAT, &format);
  TIFFGetField(ctx, TIFFTAG_COMPRESSION, &compression);
  TIFFGetField(ctx, TIFFTAG_SAMPLESPERPIXEL, &channels);
  CHECK_EQ(channels, 1);
  CHECK_EQ(bitdepth, 8);
  CHECK_EQ(format, SAMPLEFORMAT_UINT);
  CHECK_EQ(compression, COMPRESSION_NONE);
  image.Resize(Size(w, h));
  for (int r = 0; r < image.Height(); ++r) {
    TIFFReadScanline(ctx, image.Row(r), r);
  }
}

void TIFFImageReader::ReadImage(const std::string &filename, GrayscaleFloatImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "r");
  std::uint32_t w, h;
  std::uint16_t bitdepth;
  std::uint16_t format;
  std::uint16_t compression;
  std::uint16_t channels;
  TIFFGetField(ctx, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(ctx, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(ctx, TIFFTAG_BITSPERSAMPLE, &bitdepth);
  TIFFGetField(ctx, TIFFTAG_SAMPLEFORMAT, &format);
  TIFFGetField(ctx, TIFFTAG_COMPRESSION, &compression);
  TIFFGetField(ctx, TIFFTAG_SAMPLESPERPIXEL, &channels);
  CHECK_EQ(channels, 1);
  CHECK_EQ(bitdepth, 32);
  CHECK_EQ(format, SAMPLEFORMAT_IEEEFP);
  CHECK_EQ(compression, COMPRESSION_NONE);
  image.Resize(Size(w, h));
  for (int r = 0; r < image.Height(); ++r) {
    TIFFReadScanline(ctx, image.Row(r), r);
  }
}

void TIFFImageWriter::WriteImage(const std::string &filename, const RGBImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "w");
  TIFFSetField(ctx, TIFFTAG_IMAGEWIDTH, image.Width()); 
  TIFFSetField(ctx, TIFFTAG_IMAGELENGTH, image.Height()); 
  TIFFSetField(ctx, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(Byte)); 
  TIFFSetField(ctx, TIFFTAG_SAMPLESPERPIXEL, 3); 
  TIFFSetField(ctx, TIFFTAG_ROWSPERSTRIP, 1);   
  TIFFSetField(ctx, TIFFTAG_EXTRASAMPLES, EXTRASAMPLE_UNSPECIFIED); 
  TIFFSetField(ctx, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(ctx, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(ctx, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(ctx, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  TIFFSetField(ctx, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  const std::size_t bytes_per_row = 3 * image.Width() * sizeof(Byte);
  std::vector<Vector3f> cache(bytes_per_row);
  for (int r = 0; r < image.Height(); r++) {
    std::memcpy(cache.data(), image.Row(r), bytes_per_row);
    TIFFWriteScanline(ctx, cache.data(), r, 0);
  }
  TIFFClose(ctx);
}

void TIFFImageWriter::WriteImage(const std::string &filename, const RGBFloatImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "w");
  TIFFSetField(ctx, TIFFTAG_IMAGEWIDTH, image.Width()); 
  TIFFSetField(ctx, TIFFTAG_IMAGELENGTH, image.Height()); 
  TIFFSetField(ctx, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(Float)); 
  TIFFSetField(ctx, TIFFTAG_SAMPLESPERPIXEL, 3); 
  TIFFSetField(ctx, TIFFTAG_ROWSPERSTRIP, 1);   
  TIFFSetField(ctx, TIFFTAG_EXTRASAMPLES, EXTRASAMPLE_UNSPECIFIED); 
  TIFFSetField(ctx, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(ctx, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(ctx, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(ctx, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(ctx, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  const std::size_t bytes_per_row = 3 * image.Width() * sizeof(Float);
  std::vector<Vector3f> cache(bytes_per_row);
  for (int r = 0; r < image.Height(); r++) {
    std::memcpy(cache.data(), image.Row(r), bytes_per_row);
    TIFFWriteScanline(ctx, cache.data(), r, 0);
  }
  TIFFClose(ctx);
}


void TIFFImageWriter::WriteImage(const std::string &filename, const GrayscaleImage &image) {
  (void)filename;
  (void)image;
  throw NoImageSpecificationSupport("");
}

void TIFFImageWriter::WriteImage(const std::string &filename, const GrayscaleFloatImage &image) {
  TIFF *ctx = TIFFOpen(filename.c_str(), "w");
  TIFFSetField(ctx, TIFFTAG_IMAGEWIDTH, image.Width()); 
  TIFFSetField(ctx, TIFFTAG_IMAGELENGTH, image.Height()); 
  TIFFSetField(ctx, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(Float)); 
  TIFFSetField(ctx, TIFFTAG_SAMPLESPERPIXEL, 1); 
  TIFFSetField(ctx, TIFFTAG_ROWSPERSTRIP, 1);   
  TIFFSetField(ctx, TIFFTAG_EXTRASAMPLES, EXTRASAMPLE_UNSPECIFIED); 
  TIFFSetField(ctx, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(ctx, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(ctx, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(ctx, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(ctx, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  const std::size_t bytes_per_row = 1 * image.Width() * sizeof(Float);
  std::vector<Vector3f> cache(bytes_per_row);
  for (int r = 0; r < image.Height(); r++) {
    std::memcpy(cache.data(), image.Row(r), bytes_per_row);
    TIFFWriteScanline(ctx, cache.data(), r, 0);
  }
  TIFFClose(ctx);
}

}
