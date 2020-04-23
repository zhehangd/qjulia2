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

Image ReadPNGImage(std::string filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  CHECK(fp);
  
  png_byte header[8];
  CHECK_EQ(8, fread(header, 1, 8, fp));
  CHECK (png_sig_cmp(header, 0, 8) == 0);
  
  auto* png_ptr = png_create_read_struct(
    PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  CHECK(png_ptr);
  
  auto *info_ptr = png_create_info_struct(png_ptr);
  CHECK(info_ptr);
  
  int setjmp_ret = setjmp(png_jmpbuf(png_ptr));
  CHECK(setjmp_ret == 0);
  
  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  CHECK_EQ(bit_depth, 8);
  CHECK_EQ(color_type, PNG_COLOR_TYPE_RGB);
  
  int number_of_passes = png_set_interlace_handling(png_ptr);
  (void)number_of_passes;
  png_read_update_info(png_ptr, info_ptr);
  
  setjmp_ret = setjmp(png_jmpbuf(png_ptr));
  CHECK(setjmp_ret == 0);
  
  Image image(width, height);
  
  auto btype_per_row = png_get_rowbytes(png_ptr, info_ptr);
  CHECK((int)btype_per_row == width * 3);
  auto *data_ptr = image.Data()->vals;
  
  std::vector<png_bytep> row_ptrs(height);
  for (int r = 0; r < height; ++r) {row_ptrs[r] = data_ptr + r * btype_per_row;}
  
  png_read_image(png_ptr, row_ptrs.data());
  png_read_end(png_ptr, info_ptr);
  fclose(fp);
  
  return std::move(image);
}

void WritePNGImage(std::string filename, Image &image) {
  FILE *fp = fopen(filename.c_str(), "wb");
  CHECK_NOTNULL(fp);
  
  auto* png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  CHECK_NOTNULL(png_ptr);

  auto* info_ptr = png_create_info_struct(png_ptr);
  CHECK_NOTNULL(info_ptr);
  CHECK_EQ(setjmp(png_jmpbuf(png_ptr)), 0);

  png_init_io(png_ptr, fp);
  CHECK_EQ(setjmp(png_jmpbuf(png_ptr)), 0);

  png_byte bit_depth = 8;
  png_byte color_type = PNG_COLOR_TYPE_RGB;
  
  int width = image.Width();
  int height = image.Height();
  
  png_set_IHDR(png_ptr, info_ptr, width, height,
                bit_depth, color_type, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  png_write_info(png_ptr, info_ptr);
  CHECK_EQ(setjmp(png_jmpbuf(png_ptr)), 0);

  auto *data_ptr = image.Data()->vals;
  std::vector<png_bytep> row_ptrs(height);
  for (int r = 0; r < height; ++r) {row_ptrs[r] = data_ptr + r * width * 3;}
  png_write_image(png_ptr, row_ptrs.data());
  CHECK_EQ(setjmp(png_jmpbuf(png_ptr)), 0);
  png_write_end(png_ptr, NULL);
  
  fclose(fp);
}

}
