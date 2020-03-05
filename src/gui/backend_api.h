#ifndef QJULIA_ENGINE_INTERFACE_H_
#define QJULIA_ENGINE_INTERFACE_H_

#include <opencv2/opencv.hpp>

class RenderEngineInterface {
 public:
  virtual ~RenderEngineInterface(void) {};
  
  virtual void SetValue(float v) = 0;
  
  virtual cv::Size GetSize(void) const = 0;
  
  virtual cv::Mat& Render(void) = 0;
  
  virtual cv::Mat& Preview(void) = 0;
};

#endif
