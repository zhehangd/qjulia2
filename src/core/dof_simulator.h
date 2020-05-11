#ifndef QJULIA_DOF_SIMULATOR_H_
#define QJULIA_DOF_SIMULATOR_H_

#include "core/image.h"

#include <functional>

namespace qjulia {

struct MetaPixel {
  
  Vector3f exposure;
  
  Float blur_radius;
  
  // Depth of the pixel
  Float depth;
  
  // The gain of the light contribution of this pixel.
  // This is to compensate the neighborhood of the point
  // that is occulded by the foreground.
  Float compensation = 0;
  
  // The opacity of light coming behind the pixel
  //Float opacity_near = 0;
  
  Float opacity_far = 0;
};

class BlurKernel {
 public:
  
  // Initialize the kernel, given the maximum blur radius needed
  // with this kernel.
  //void Init(Float max_blur_radius);
  
  /// @brief Returns the half of the kernel size needed for the given blur radius
  CPU_AND_CUDA static int CalculateHalfKernelSize(float radius);
  
  /// @brief Returns the size of the kernel needed for the given blur radius
  ///
  /// Always an odd, positive number
  CPU_AND_CUDA static int CalculateKernelSize(float radius);
  
  /// @brief Sample the coefficient of the kernel at the given coordinates
  ///
  /// (0, 0) is considered the center of the kernel
  CPU_AND_CUDA static Float Sample(Float x, Float y, Float radius);
};

class DOFSimulator {
 public:
  
  void SetDepthPrecision(Float val) {data_.depth_precision = val;}
  
  Float GetDepthPrecision(void) const {return data_.depth_precision;}
  
  void SetBlurStrength(Float val) {data_.blur_strength = val;}
  
  Float GetBlurStrength(void) const {return data_.blur_strength;}
  
  void SetFarDOFLimit(Float val) {data_.far_dof_limit = val;}
  
  Float GetFarDOFLimit(void) const {return data_.far_dof_limit;}
  
  void SetFarHyperfocalDepth(Float val) {data_.far_hyperfocal_depth = val;}
  
  Float GetFarHyperfocalDepth(void) const {return data_.far_hyperfocal_depth;}
  
  void SetOcclusionAttenuation(Float val) {data_.occlusion_attenuation = val;}
  
  Float GetOcclusionAttenuation(void) const {return data_.far_hyperfocal_depth;}
  
  void Parse(const Args &args);
   
  virtual void Process(RGBFloatImage exposure, GrayscaleFloatImage depth,
                       RGBFloatImage &dst) = 0;
  
  struct Data {
    int width;
  
    int height;
    
    int minwh;
    
    Float max_blur_raidus;
    
    // Tolerance in various depth comparison
    Float depth_precision = 0.1;
    
    // How strong the blur is
    Float blur_strength = 1.0;
    
    // Far DOF limits
    Float far_dof_limit = 1.0;
    
    // Depth where the blur strength reaches the maximum
    Float far_hyperfocal_depth = 2.0;
    
    Float occlusion_attenuation = 0.0;
    
    int max_receptive_field_size;
    
    Array2D<MetaPixel> cache_meta; // = short_ * blur_strength_ / 100.0
  };
 
 protected:
  
  Data data_;
};

class DOFSimulatorCPU : public DOFSimulator {
 public:
  
  void Process(RGBFloatImage exposure, GrayscaleFloatImage depth,
                       RGBFloatImage &dst) override;
 private:
  
  void MultithreadCall(std::function<void(int,int)> fn);
  
  void WindowCall(int curr_r, int curr_c, Float blur_radius, std::function<void(int,int)> fn);
  
  Float DepthToBlurRadius(Float depth);
  
  Float StrengthToBlurRadius(Float strength);
  
  void Analyze(RGBFloatImage exposure, GrayscaleFloatImage depth);
  
  void Compose(RGBFloatImage &dst);
};

}

#endif
