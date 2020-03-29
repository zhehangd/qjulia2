#ifndef QJULIA_ENGINE_INTERFACE_H_
#define QJULIA_ENGINE_INTERFACE_H_

#include <opencv2/opencv.hpp>

class RenderEngineInterface {
 public:
  
  struct SceneOptions {
    float julia_constant[4] {-0.2,0.8,0,0};
    float camera_pose[3] {10, 0, 5.3}; // azimuth/altitude/distance
  };
   
  virtual ~RenderEngineInterface(void) {};
  
  virtual void SetValue(float v) = 0;
  
  virtual cv::Size GetSize(void) const = 0;
  
  virtual cv::Mat Render(SceneOptions options) = 0;
  
  virtual cv::Mat Preview(SceneOptions options) = 0;
};

#endif
