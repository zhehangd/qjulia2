#ifndef SCENE_CTRL_PARAMS_H_
#define SCENE_CTRL_PARAMS_H_

#include "core/array2d.h"
#include "core/vector.h"

#include <string>

struct SceneCtrlParams {
  qjulia::Size realtime_image_size;
  qjulia::Size realtime_fast_image_size;
  qjulia::Size offline_image_size;
  std::string offline_filename;
};

#endif
