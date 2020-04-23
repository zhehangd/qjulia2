#ifndef QJULIA_INTEGRATOR_RETURN_H_
#define QJULIA_INTEGRATOR_RETURN_H_

namespace qjulia {

/// @brief Data returned by integrators
///
/// An Sample stores the data collected by
/// an integrator, through a ray passing through the scene.
/// It is the responsibility 
struct Sample {
  
  // Light collected by the ray
  Spectrum spectrum;
  
  // Depth of the first intersection
  Float depth;
  
  // If the ray has made an intersection
  bool has_isect;
};

}

#endif
