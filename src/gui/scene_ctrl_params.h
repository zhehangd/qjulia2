#ifndef SCENE_CTRL_PARAMS_H_
#define SCENE_CTRL_PARAMS_H_

#include "core/array2d.h"
#include "core/vector.h"

#include <string>

struct SceneCtrlParams {
  qjulia::Quaternion fractal_constant;
  float fractal_precision;
  bool fractal_cross_section;
  float fractal_uv_black;
  float fractal_uv_white;
  
  qjulia::Size realtime_image_size;
  qjulia::Size realtime_fast_image_size;
  qjulia::Size offline_image_size;
  std::string offline_filename;
  
  qjulia::Vector3f camera_target;
  qjulia::Vector3f camera_pose; // azimuth/altitude/distance
  float camera_fov;
  float camera_headlight_lumin;
};

#endif
