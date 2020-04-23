/*

MIT License

Copyright (c) 2019 Zhehang Ding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <gtest/gtest.h> 

#include "core/array2d.h" 
#include "core/texture.h" 
#include "core/image.h"

using namespace qjulia;

#ifdef WITH_CUDA

__global__ void NewTextureKernel(Texture **tex_device) {
  *tex_device = new Texture();
};

__global__ void DeleteTextureKernel(Texture **tex_device) {
  delete *tex_device;
};

__global__ void SampleTexture(Texture *tex, int num_samples,
                            Vector2f *coords, Vector3f *colors) {
  for (int i = 0; i < num_samples; ++i) {
    colors[i] = tex->At(coords[i]);
  }
}

TEST(Texture, SamplingCUDA) {
  int num_samples = 8;
  Vector2f *sample_coords;
  Vector3f *sample_colors;
  Texture **tex_device;
  cudaMallocManaged(&tex_device, sizeof(Texture*));
  cudaMallocManaged(&sample_coords, num_samples * sizeof(Vector2f));
  cudaMallocManaged(&sample_colors, num_samples * sizeof(Vector3f));
  cudaDeviceSynchronize();
  
  for (int i = 0; i < num_samples; ++i) {
    sample_coords[i] = {(float)i / (num_samples - 1), 0};
  }
  
  NewTextureKernel<<<1, 1>>>(tex_device);
  
  Image image(Size(8, 1));
  image.At(0, 0) = {255, 128, 0};
  image.At(0, 1) = {64, 10, 100};
  image.At(0, 2) = {255, 128, 50};
  image.At(0, 3) = {255, 0, 255};
  image.At(0, 4) = {124, 255, 1};
  image.At(0, 5) = {255, 0, 208};
  image.At(0, 6) = {0, 12, 12};
  image.At(0, 7) = {255, 255, 200};
  
  Texture tex;
  tex.LoadImage(image);
  tex.UpdateDevice(*tex_device);
  SampleTexture<<<1, 1>>>(*tex_device, num_samples, sample_coords, sample_colors);
  cudaDeviceSynchronize();
  
  for (int i = 0; i < num_samples; ++i) {
    Vector2f coords = sample_coords[i];
    int c = static_cast<int>(image.Width() * coords[0]);
    if (c < 0) {c = 0;}
    if (c >= image.Width()) {c = image.Width() - 1;}
    Vector3b c3b = image.At(0, c);
    EXPECT_NEAR(sample_colors[i][0], c3b[0] / 255.0f, 1e-3);
    EXPECT_NEAR(sample_colors[i][1], c3b[1] / 255.0f, 1e-3);
    EXPECT_NEAR(sample_colors[i][2], c3b[2] / 255.0f, 1e-3);
  }
  
  DeleteTextureKernel<<<1, 1>>>(tex_device);
  cudaFree(tex_device);
  cudaFree(sample_coords);
  cudaFree(sample_colors);
}

#endif
