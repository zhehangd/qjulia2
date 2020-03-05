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

#include <cassert>
#include <fstream>
#include <string>


#include "core/messages.h"
#include "core/resource_mgr.h"
#include "core/scene.h"

namespace qjulia {

bool ResourceMgr::RegisterPrototypeEntity(SceneEntity *entity) {
  int eid = static_cast<int>(entity->GetType());
  std::string name = entity->GetImplName();
  auto &prototype_map = prototype_entities_[eid];
  auto it = prototype_map.find(name);
  if (it == prototype_map.end()) {
    auto ret = prototype_map.emplace(
      name, std::unique_ptr<SceneEntity>(entity));
    if (ret.second) {
      return true;
    } else {
      std::cerr << "Error: unknown error, cannot register \""
        << name << "\"." << std::endl;
      return false;
    }
  } else {
    std::cerr << "Error: template " << name << " already exists." << std::endl;
    return false;
  }
}

Scene* ResourceMgr::GetScene(void) const {
  const int eid = GetEntityTypeID(EntityType::kScene);
  const auto &entity_map = entity_maps_[eid];
  if (entity_map.size() == 0) {return nullptr;}
  return dynamic_cast<Scene*>(entity_map.begin()->second.get());
}

bool ResourceMgr::LoadSceneDescription(
    const std::string &filename) {
  std::vector<TokenizedBlock> blocks;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Cannot open the file \"" << filename << "\".";
    return false;
  }
  TokenizeSceneFile(file, &blocks);
  for (const auto &block : blocks) {
    bool good = ParseFromTokenizedBlock(block);
    if (!good) {return false;}
  }
  return true;
}

void ResourceMgr::PrintPrototypes(std::ostream &os) const {
  for (const auto &prototype_map : prototype_entities_) {
    for (const auto &prototype_pair : prototype_map) {
      const auto &prototype = prototype_pair.second;
      os << GetEntityTypeName(prototype->GetType()) << " - "
        << prototype->GetImplName() << std::endl;
    }
  }
}

void ResourceMgr::PrintEntities(std::ostream &os) const {
  for (const auto &entity_map : entity_maps_) {
    for (const auto &prototype_pair : entity_map) {
      const auto &entity = prototype_pair.second;
      os << GetEntityTypeName(entity->GetType()) << " - "
        << entity->GetImplName() << " - " << prototype_pair.first << std::endl;
    }
  }
}

bool ResourceMgr::ParseFromTokenizedBlock(const TokenizedBlock &block) {
  const std::string &name = block.name;
  const std::string &type = block.type;
  const std::string &impl = block.subtype;
  // Find type id from the type name.
  int eid = 0;
  for (eid = 0; eid < kNumEntityTypes; ++eid) {
    if (std::string(kEntityTypeNames[eid]) == type) {break;}
  }
  if (eid >= kNumEntityTypes) {
    std::cerr << "Error: unknown entity type \"" << type << "\"." << std::endl;
    return false;
  }
  // Find if the name maps to an existing entity.
  auto &entity_map = entity_maps_[eid];
  auto it = entity_map.find(name);
  // If does not exist, find the corresponding prototype and make a clone.
  if (it == entity_map.end()) {
    auto &prototype_map = prototype_entities_[eid];
    auto prototype_it = prototype_map.find(impl);
    if (prototype_it == prototype_map.end()) {
      std::cerr << "Error: no prototype for \"" << impl << "\" found." << std::endl;
    }
    SceneEntity *entity = prototype_it->second->Clone();
    auto ret = entity_map.emplace(name, std::unique_ptr<SceneEntity>(entity));
    if (!(ret.second)) {
      std::cerr << "Error: cannot add \""<< name << "\"." << std::endl;
      return false;
    }
    it = ret.first;
  }
  for (const auto &instruction : block.instructions) {
    if (!it->second->ParseInstruction(instruction, this)) {
      std::cerr << "Error: cannot parse \""
        << TokenizedStatement2Str(instruction) << "\"." << std::endl;
      // just skip it, do not stop.
    }
  }
  return true;
}

}
