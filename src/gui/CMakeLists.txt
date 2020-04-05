
add_executable(qjulia2-gui
  mainwindow.cc
  camerapanel.cc
  controlbar.cc
  fractalpanel.cc
  generalpanel.cc
  main.cc
  engine.cc
  $<TARGET_OBJECTS:qjulia2-core>
)

# Let generated UI files see the headers
target_include_directories(qjulia2-gui PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(qjulia2-gui
  Qt5::Widgets
  Qt5::Concurrent
  qjulia2-core
)

set_target_properties(qjulia2-gui
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
