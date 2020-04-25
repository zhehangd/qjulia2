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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "core/qjulia2.h"
#include "core/arg_parse.h"
#include "core/image.h"

using namespace qjulia;

Vector2i ParseImageSize(std::string &size_str) {
  int x_i = size_str.find('x');
  int w = std::stoi(size_str.substr(0, x_i));
  int h = std::stoi(size_str.substr(x_i + 1));
  return {w, h};
}

bool Run(int argc, char **argv) {
  
  cxxopts::Options cxxopts_options("qjulia", "Quaternion Julia Set Renderer");
  
  cxxopts_options.add_options()
  ("a,antialias", "Antialiasing mode {0, 1, 2, 3}",\
    cxxopts::value<int>()->default_value("1"))
  ("t,num_threads", "Number of threads",\
    cxxopts::value<int>()->default_value("-1"))
  ("s,size", "Output image size",
    cxxopts::value<std::string>()->default_value("640x360"))
  ("o,output_file", "Output image filename",
    cxxopts::value<std::string>()->default_value("output.png"))
  ("i,scene_file", "Input scene filename",
    cxxopts::value<std::string>()->default_value(""))
  ;
  
  auto args = cxxopts_options.parse(argc, argv);
  int num_threads = args["num_threads"].as<int>();
  std::string size_str = args["size"].as<std::string>();
  std::string output_file = args["output_file"].as<std::string>();
  std::string scene_file = args["scene_file"].as<std::string>();
  
  if (scene_file.empty()) {
    std::cerr << "Use \"-i [FILE]\" to specify "
      "the scene description file." << std::endl;
    return false;
  }
  
  SceneBuilder build;
  RegisterDefaultEntities(build);
  
  SceneDescr scene_descr = LoadSceneFromFile(scene_file);
  build.ParseSceneDescr(scene_descr);
  
  //SceneDescr scene_descr_out = build.SaveSceneDescr();
  //SaveSceneDescrToFile("out.scene", scene_descr_out);
  
  RenderOptions option;
#ifdef WITH_CUDA
  option.cuda = true;
#else
  option.cuda = false;
#endif
  option.aa = static_cast<AAOption>(args["antialias"].as<int>());
  option.num_threads = num_threads;
  
  Vector2i size = ParseImageSize(size_str);
  if (size[0] <= 0 || size[1] <= 0) {
    std::cerr << "Error: Invalid image size "
      << size[0] << "x" << size[1] << "." << std::endl;
    return false;
  }
  
  Image image(Size(size[0], size[1]));  
  RTEngine engine;
  engine.Render(build, option, image);
  LOG(INFO) << "Rendering time: " << engine.LastRenderTime();
  
  WritePNGImage(output_file, image);
  return true;
}

int main(int argc, char **argv) {
  google::SetStderrLogging(google::GLOG_INFO);
  google::InitGoogleLogging(argv[0]);
  
  bool good = Run(argc, argv);
  return good ? 0 : 1;
}
