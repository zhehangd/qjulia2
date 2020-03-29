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

#include "scene_builder.h"

#include "core/entity.h"
#include "core/scene_builder_register.h"

#include "core/material.h"
#include "core/object.h"
#include "core/scene.h"
#include "core/world.h"
#include "core/transform.h"

#include "core/camera/camera3d.h"
#include "core/light/simple.h"
#include "core/shape/julia3d.h"
#include "core/shape/plane.h"
#include "core/shape/sphere.h"
#include "core/scene_descr.h"

namespace qjulia {

std::string SceneBuilder::GetSTypeName(size_t stype_id) const {
  return reg_table_[stype_id].stype_name;
}

void SceneBuilder::ParseSceneDescr(const SceneDescr &descr) {
  for (const auto &e : descr.entities) {
    ParseEntityDescr(e);
  }
}
  
void SceneBuilder::ParseEntityDescr(const EntityDescr &descr) {
  EntityNode *node = CreateEntity(descr.type, descr.subtype, descr.name);
  CHECK_NOTNULL(node);
  Entity *entity = node->Get();
  CHECK_NOTNULL(node);
  for (const auto &statement : descr.statements) {
    entity->Parse(statement, this);
  }
}
  
EntityNode* SceneBuilder::CreateEntity(
    std::string btype, std::string stype, std::string name) {
  if (btype == EntityTrait<Camera>::name) {
    return CreateEntity<Camera>(stype, name);
  } else if (btype == EntityTrait<Light>::name) {
    return CreateEntity<Light>(stype, name);
  } else if (btype == EntityTrait<Material>::name) {
    return CreateEntity<Material>(stype, name);
  } else if (btype == EntityTrait<Object>::name) {
    return CreateEntity<Object>(stype, name);
  } else if (btype == EntityTrait<Shape>::name) {
    return CreateEntity<Shape>(stype, name);
  } else if (btype == EntityTrait<Transform>::name) {
    return CreateEntity<Transform>(stype, name);
  } else if (btype == EntityTrait<World>::name) {
    return CreateEntity<World>(stype, name);
  } else {
    throw UnknownBTypeExcept(btype);
  }
}

void SceneBuilder::DebugPrint(void) const {
  LOG(INFO) << "Registered records:";
  for (const auto &record : reg_table_) {
    LOG(INFO) << "Record:";
    LOG(INFO) << "stype_name: " << record.stype_name;
    LOG(INFO) << "btype_id: " << record.btype_id;
    LOG(INFO) << "stype_id: " << record.stype_id;
    LOG(INFO) << "-----------------";
  }
  LOG(INFO) << "Created nodes:";
  for (auto &node : nodes_) {
    LOG(INFO) << "Node:";
    LOG(INFO) << "Name: " << node->GetName();
    LOG(INFO) << "BType: " << node->btype_id_;
    LOG(INFO) << "SType: " << node->stype_id_;
    LOG(INFO) << "-----------------";
  }
}

Scene SceneBuilder::BuildScene(const BuildSceneParams &params) const {
  // TODO: How to make sure only cameras in this world are searched?
#ifndef WITH_CUDA
  if (params.cuda) {
    LOG(ERROR) << "qjulia2 is not compiled with CUDA, fallback to CPU mode";
    params.cuda = false;
  }
#endif
  Scene scene;
  if (params.cuda) {
    for (auto &node : nodes_) {
      CHECK_NOTNULL(node->Get())->UpdateDevice(
        CHECK_NOTNULL(node->GetDevice()));
    }
    auto *node_world = SearchEntityByName<World>(params.world);
    scene.world_ = node_world->GetDevice();
    auto *node_camera = SearchEntityByName<Camera>(params.camera);
    scene.camera_ = node_camera->GetDevice();
  } else {
    auto *node_world = SearchEntityByName<World>(params.world);
    scene.world_ = node_world->Get();
    auto *node_camera = SearchEntityByName<Camera>(params.camera);
    scene.camera_ = node_camera->Get();
  }
  return scene;
}

void RegisterDefaultEntities(SceneBuilder &build) {
  build.Register<Object>("");
  build.Register<Material>("");
  build.Register<Transform>("");
  build.Register<World>("");
  build.Register<PerspectiveCamera>("Perspective");
  build.Register<OrthoCamera>("Ortho");
  build.Register<PointLight>("Point");
  build.Register<SunLight>("Sun");
  build.Register<Julia3DShape>("Julia3D");
  build.Register<PlaneShape>("Plane");
  build.Register<SphereShape>("Sphere");
}

}

