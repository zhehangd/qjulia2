add_subdirectory(widgets)

add_executable(qjulia2-gui
  mainwindow.cc
  panel.cc
  generalpanel.cc
  
  panel_dev.cc
  module_base.cc
  module_camera.cc
  module_placeholder.cc
  module_sun_light.cc
  module_point_light.cc
  module_julia3d_shape.cc
  
  main.cc
  qjulia_context.cc
  $<TARGET_OBJECTS:qjulia2-gui-widgets>
  $<TARGET_OBJECTS:qjulia2-core>
)

# Let generated UI files see the headers
target_include_directories(qjulia2-gui PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(qjulia2-gui
  Qt5::Widgets
  Qt5::Concurrent
  qjulia2-gui-widgets
  qjulia2-core
)

set_target_properties(qjulia2-gui
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
