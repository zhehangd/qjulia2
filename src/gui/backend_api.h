#ifndef QJULIA_ENGINE_INTERFACE_H_
#define QJULIA_ENGINE_INTERFACE_H_

#include <opencv2/opencv.hpp>

class RenderEngineInterface {
 public:
  

   
  virtual ~RenderEngineInterface(void) {};
  
  virtual cv::Size GetSize(void) const = 0;
  
  virtual cv::Mat Render(SceneOptions options) = 0;
  
  virtual cv::Mat Preview(SceneOptions options) = 0;
};

#endif
