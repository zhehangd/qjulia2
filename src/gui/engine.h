#ifndef QJULIA_GUI_ENGINE_H_
#define QJULIA_GUI_ENGINE_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "core/qjulia2.h"

class RenderEngine {
 public:
   
  struct SceneOptions {
    float julia_constant[4] {0,0,0,0};
    float camera_pose[3] {10, 0, 5.3}; // azimuth/altitude/distance
  };
   
  RenderEngine(void);
  ~RenderEngine(void);
  
  void Init(std::string scene_file);
  
  cv::Size GetSize(void) const;
  
  cv::Mat Render(SceneOptions options);
  
  cv::Mat Preview(SceneOptions options);
  
 private:
   
  class Impl;
  std::unique_ptr<Impl> impl_;
};


#endif
