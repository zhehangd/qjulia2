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

#ifndef QJULIA_DEVELOPER_H_
#define QJULIA_DEVELOPER_H_

#include "base.h"
#include "dof_simulator.h"
#include "film.h"
#include "image.h"
#include "vector.h"
#include "entity.h"

namespace qjulia {

struct CachePixel {
  Spectrum spectrum;
  Float depth;
  Float w = 0;
  Float depth_w = 0;
};

struct DevOptions {
  
  bool flag_save_exposure_map = false;
  
  bool flag_save_depth_map = false;
  
  bool flag_use_dof = false;
  
  std::string exposure_map_filename;
  
  std::string depth_map_filename;
  
};

/// @brief A Developer process a film into an image
///
/// After an rendering engine uses an integrator to produce a film
/// which contains more or less physically based information of each
/// pixel, a developer is responsible for making an image from the
/// information in the film.
class Developer {
 public:
   
  /// @brief Informs the developer that a new rendering begins
  void Init(Size size);
  
  /// @brief Informs the devloper that the rendering is done
  void Finish(void);
  
  /// @brief Save the raw exposure map to the file
  /// 
  /// If filename is not empty, the raw exposure map will be saved to the file
  /// once the rendering is done. If filename is empty, will not save anything.
  void SetExposureExport(std::string filename);
  
  /// @brief Save the raw depth map to the file
  ///
  /// If filename is not empty, the raw depth map is saved to the file
  /// once the rendering is done. If filename is empty, will not save anything.
  void SetDepthExport(std::string filename);
  
  void EnableDOF(bool enable);
  
  void ProduceImage(RGBImage &image);
  
  void Parse(const Args &args);
  
  DOFSimulator& GetDOFSimulator(void) {return dof_;}
  
 protected:
   
  void ProcessSampleFrameImpl(SampleFrame &film, float w);
  
  Array2D<CachePixel> cache_;
  
  RGBFloatImage exposure_cache_;
  GrayscaleFloatImage depth_cache_;
  
  DevOptions options_;
  
  DOFSimulatorCPU dof_;
};

class DeveloperGPU : public Developer {
 public:
  void ProcessSampleFrame(SampleFrame &film, float w);
};

class DeveloperCPU : public Developer {
 public:
  void ProcessSampleFrame(SampleFrame &film, float w);
};

}

#endif
