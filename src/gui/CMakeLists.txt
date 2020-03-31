
add_executable(qjulia2-gui
  mainwindow.cc
  main.cc
  engine.cc
  $<TARGET_OBJECTS:qjulia2-core>
)

target_link_libraries(qjulia2-gui
  Qt5::Widgets
  Qt5::Concurrent
  qjulia2-core
)

set_target_properties(qjulia2-gui
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
