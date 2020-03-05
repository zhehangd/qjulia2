
add_executable(qjulia2-gui
  mainwindow.cc
  main.cc
  engine.cc
  $<TARGET_OBJECTS:qjulia2-core>
)

target_link_libraries(qjulia2-gui
  Qt5::Widgets
  ${OpenCV_LIBS}
  qjulia2-core
)
