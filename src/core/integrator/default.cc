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

CPU_AND_CUDA Vector3f MakeReflectionVector(const Vector3f &vin, const Vector3f &normal) {
  return normal * (2 * Dot(vin, normal)) - vin;
}

CPU_AND_CUDA Sample DefaultIntegrator::Li(const Ray &ray, const Scene &scene) {
  return LiRecursive(ray, scene, 1);
}

CPU_AND_CUDA Sample DefaultIntegrator::LiRecursive(
    const Ray &ray, const Scene &scene, int fork_depth) {
  
  Sample sample;
  
  // Accumulate the light from various sources
  Spectrum final_spectrum;
  
  // Find intersection.
  Intersection isect;
  const Object* hit_object = scene.Intersect(ray, &isect);
  if (hit_object == nullptr) {return sample;}
  
  const Material *material = hit_object->GetMaterial();
  const Texture *texture = material->GetTexture();
  
  // Light illumination
  for (int i = 0; i < scene.NumLights(); ++i) {
    // Compute the incident ray from a light.
    const Light *light = scene.GetLight(i);
    LightRay lray = light->Li(isect.position);    
    
    // Cosine of the incident ray.
    // This is used to decide whether the light shines the front
    // face of the surface, and decide the illuminance on the
    // surface.
    Float in_cosine = Dot(lray.wi, isect.normal);
    if (in_cosine <= 0) {continue;}
    
    // Test if the light is occluded
    auto occ_start = isect.position + isect.normal * ray_delta_;
    auto occ_dir = lray.wi;
    Ray occ_ray(occ_start, occ_dir);
    Intersection occ_isect;
    const Object *occ_object = scene.Intersect(occ_ray, &occ_isect);
    (void)occ_object;
    if (occ_isect.good && occ_isect.dist <= lray.dist) {
      continue;
    }
    
    // Diffusion
    Spectrum spec_diffuse = lray.spectrum * material->diffuse * in_cosine;
    if (texture) {spec_diffuse *= texture->At(isect.uv);}
    sample.spectrum += spec_diffuse;
    
    // Specular
    Float specular_cosine = Dot(-ray.dir, MakeReflectionVector(lray.wi, isect.normal));
    specular_cosine = specular_cosine > 0 ? specular_cosine : 0;
    specular_cosine = std::pow(specular_cosine, material->ps);
    Spectrum spec_specular = lray.spectrum * material->ks * specular_cosine;
    sample.spectrum += spec_specular;
  }
  
  // Reflection
  if (material->reflection > 0 && fork_depth > 0) {
    auto ref_dir = MakeReflectionVector(-ray.dir, isect.normal);
    auto ref_start = isect.position + ref_dir * ray_delta_;
    Ray ref_ray(ref_start, ref_dir);
    Sample ref_sample = LiRecursive(ref_ray, scene, fork_depth - 1);
    ref_sample.spectrum *= material->reflection;
    sample.spectrum += ref_sample.spectrum;
  }
  
  sample.depth = Dist(isect.position, ray.start);
  sample.has_isect = true;
  return sample;
}

}

