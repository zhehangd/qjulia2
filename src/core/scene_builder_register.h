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

#ifndef QJULIA_SCENE_BUILD_REGISTER_H_
#define QJULIA_SCENE_BUILD_REGISTER_H_

#include "scene_builder.h"

#include <exception>
#include <memory>
#include <string>

#if !defined(WITH_CUDA) || defined(__CUDACC__)

#if defined(WITH_CUDA) && !defined(__CUDACC__)
#error If CUDA is enabled, any file that includes this must be compiled with NVCC
#endif

namespace qjulia {

#ifdef __CUDACC__
/// @brief Device kernel to create a new entity object
template<typename ST>
KERNEL void AllocateDeviceKernel(ST **dst) {
  *dst = new ST();
}
#endif

/// @brief EntityNode for a specific type
template <typename ST>
struct EntityNodeST : public EntityNodeBT<typename EntityTrait<ST>::BaseType> {
  
  using BaseType = typename EntityTrait<ST>::BaseType;
  
  EntityNodeST(void);
  
  // Returns the pointer to the entity
  ST* Get(void) override {return host_ptr_.get();}
  
  // Returns the pointer to the entity on the device (GPU)
  ST* GetDevice(void) override {return device_ptr_.get();}
  
  void AllocateHost(void) override {host_ptr_.reset(new ST());}
  
  void ReleaseHost(void) override {host_ptr_.reset(nullptr);}
  
  void AllocateDevice(void) override {
    ST *p = nullptr;
#ifdef __CUDACC__
    ST **dpp = nullptr;
    cudaMalloc((void **)&dpp, sizeof(ST*));
    AllocateDeviceKernel<ST><<<1, 1>>>(dpp);
    cudaMemcpy(&p, dpp, sizeof(ST*), cudaMemcpyDeviceToHost);
    cudaFree(dpp);
#endif
    device_ptr_.reset(p);
  }
  
  void ReleaseDevice(void) override {
    device_ptr_.reset(nullptr);
  }
  
  void UpdateDevice(void) override {
    Get()->UpdateDevice(GetDevice());
  }
  
#ifdef __CUDACC__
  std::unique_ptr<ST, void(*)(ST*)> device_ptr_
    {nullptr, [](ST *p) {cudaFree(p);}};
#else
  std::unique_ptr<ST> device_ptr_ {};
#endif
  
  std::unique_ptr<ST> host_ptr_;
};

template <typename ST>
EntityNodeST<ST>::EntityNodeST(void) {
  AllocateHost();
  AllocateDevice();
}

template <typename ST>
bool SceneBuilder::Register(std::string stype_name) {
  auto btype_id = EntityTrait<ST>::btype_id;
  for (const auto &record : reg_table_) {
    if (record.btype_id != btype_id) {continue;}
    if (record.stype_name == stype_name) {
      throw RegisterFailedExcept(EntityTrait<ST>::name, stype_name);
    }
  }
  RegRecord record;
  record.stype_name = stype_name;
  record.btype_id = btype_id;
  record.stype_id = reg_table_.size(); // equal to the index in reg_table 
  record.fn_create = []() -> EntityNode* {return new EntityNodeST<ST>();};
  reg_table_.push_back(record);
  return true;
}

}

#endif

#endif
