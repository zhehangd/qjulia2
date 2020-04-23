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

#include "core/integrator/default.h"

#include <cmath>
#include <limits>

#include <glog/logging.h>

namespace qjulia {

CPU_AND_CUDA Vector3f ReflectVector(const Vector3f &vin, const Vector3f &normal) {
  return normal * (2 * Dot(vin, normal)) - vin;
}

CPU_AND_CUDA Sample DefaultIntegrator::Li(const Ray &ray, const Scene &scene) {
  return LiRecursive(ray, scene, 1);
}

CPU_AND_CUDA Sample DefaultIntegrator::LiRecursive(
    const Ray &ray, const Scene &scene, int depth) {
  
  Spectrum final_spectrum;
  
  // Find the nearest intersection.
  // If no intersection found, return empty spectrum.
  Intersection isect;
  const Object* hit_object = scene.Intersect(ray, &isect);
  if (hit_object == nullptr) {
    return {final_spectrum, 0};
  }
  
  const Vector3f &hit_position = isect.position;
  const Vector3f &hit_normal = isect.normal;
  
  const Material *material = hit_object->GetMaterial();
  const Texture *texture = material->GetTexture();
  
  // Light illumination
  for (int i = 0; i < scene.NumLights(); ++i) {
    // Compute the incident ray from a light.
    const Light *light = scene.GetLight(i);
    LightRay lray = light->Li(hit_position);    
    
    // Cosine of the incident ray.
    // This is used to decide whether the light shines the front
    // face of the surface, and decide the illuminance on the
    // surface.
    Float in_cosine = Dot(lray.wi, hit_normal);
    if (in_cosine <= 0) {continue;}
  
    // Test occlusion
    Ray in_ray(hit_position + hit_normal * ray_delta_, lray.wi);
    Intersection occ_isect;
    const Object *occ_object = scene.Intersect(in_ray, &occ_isect);
    (void)occ_object; // reserved
    
    // If occluded
    if (occ_isect.good) {
      if (occ_isect.dist <= lray.dist) {
        continue;
      }
    }
    
    Float reflect = Dot(-ray.dir, ReflectVector(lray.wi, hit_normal));
    reflect = reflect > 0 ? reflect : 0;
    reflect = std::pow(reflect, material->ps);
    
    Spectrum spec_diffuse = lray.spectrum * material->diffuse * in_cosine;
    if (texture) {
      spec_diffuse *= texture->At(isect.uv);
    }
    
    final_spectrum += lray.spectrum * material->ks * reflect;
    final_spectrum += spec_diffuse;
  }
  
  // Reflection
  if (material->reflection > 0 && depth > 0) {
    Vector3f ray_dir = ReflectVector(-ray.dir, hit_normal);
    Sample recret = LiRecursive(
      Ray(hit_position + ray_dir * ray_delta_, ray_dir),
      scene, depth - 1);
    recret.spectrum *= material->reflection;
    final_spectrum += recret.spectrum;
  }
  
  return Sample{final_spectrum, 0};
}

}

