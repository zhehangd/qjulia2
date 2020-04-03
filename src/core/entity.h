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

#ifndef QJULIA_ENTITY_H_
#define QJULIA_ENTITY_H_

#include "base.h"
#include <vector>
#include <string>

namespace qjulia {

typedef std::vector<std::string> Args;  

class SceneBuilder;

// TODO: handle object copy and CUDA copy

// @brief An object that can be defined in a scene file
class Entity {
 public:
  
  CPU_AND_CUDA virtual ~Entity(void) {}
  
  virtual void Parse(const Args &args, SceneBuilder *build) {
    (void)args; (void)build;
    LOG(FATAL) << "No parsing function defined"; 
  }
  
  virtual void UpdateDevice(Entity *device_ptr) const {
    (void)device_ptr;
    LOG(FATAL) << "UpdateDevice is not implemented";
  }
  
  CPU_AND_CUDA virtual void DebugPrint(void) const {}
};

// Six basic entity types
class Camera; class Light; class Material; class Object;
class Shape; class Texture; class Transform; class World;

/// @brief Gets the ID of a basic type
///
/// One can also use EntityTrait<T>::btype_id to achieve the same purpose.
template <typename T> struct EntityTypeID;
template <> struct EntityTypeID<Camera> {static const size_t val = 0;};
template <> struct EntityTypeID<Light> {static const size_t val = 1;};
template <> struct EntityTypeID<Material> {static const size_t val = 2;};
template <> struct EntityTypeID<Object> {static const size_t val = 3;};
template <> struct EntityTypeID<Shape> {static const size_t val = 4;};
template <> struct EntityTypeID<Texture> {static const size_t val = 5;};
template <> struct EntityTypeID<Transform> {static const size_t val = 6;};
template <> struct EntityTypeID<World> {static const size_t val = 7;};

/// @brief Gets the basic type from its ID
template <size_t N> struct EntityType;
template <> struct EntityType<0> {typedef Camera type;};
template <> struct EntityType<1> {typedef Light type;};
template <> struct EntityType<2> {typedef Material type;};
template <> struct EntityType<3> {typedef Object type;};
template <> struct EntityType<4> {typedef Shape type;};
template <> struct EntityType<5> {typedef Texture type;};
template <> struct EntityType<6> {typedef Transform type;};
template <> struct EntityType<7> {typedef World type;};

/// @brief Name table of the basic types
/// It is used to match of description text
constexpr const char *kEntityTypeNames[] = {
    "Camera", "Light", "Material", "Object", "Shape", "Texture", "Transform", "World"};

/// @brief Trait of an entity type
/// The template argument can be either a basic or a specific type.
/// The trait provides the info of the corresponding basic type.
template <typename T>
struct EntityTrait {
  
  using Type = T;
  
  /// @brief Basic type
  using BaseType = 
    typename std::conditional<std::is_base_of<Camera, T>::value, Camera,
    typename std::conditional<std::is_base_of<Light, T>::value, Light,
    typename std::conditional<std::is_base_of<Material, T>::value, Material,
    typename std::conditional<std::is_base_of<Object, T>::value, Object,
    typename std::conditional<std::is_base_of<World, T>::value, World,
    typename std::conditional<std::is_base_of<Shape, T>::value, Shape,
    typename std::conditional<std::is_base_of<Texture, T>::value, Texture,
    typename std::conditional<std::is_base_of<Transform, T>::value, Transform,
    void>::type>::type>::type>::type>::type>::type>::type>::type;
  
  /// @brief ID of the basic type
  static const size_t btype_id = EntityTypeID<BaseType>::val;
  
  /// @brief Name of the type
  static constexpr const char *name = kEntityTypeNames[btype_id];
};

}

#endif
