
# Find the default directory to install the plugins
execute_process(COMMAND qtpaths --plugin-dir OUTPUT_VARIABLE QT_SYSTEM_PLUGIN_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(QT_SYSTEM_PLUGIN_DESIGNER_DIR "${QT_SYSTEM_PLUGIN_DIR}/designer")

function(generate_target_name TARGETNAME BASENAME)
  set(${TARGETNAME} "${BASENAME}" PARENT_SCOPE)
endfunction()

function(check_file_exists FILENAME)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
  else()
    message(FATAL_ERROR "'${FILENAME}' does not exist")
  endif()
endfunction()

# Generates a qjulia2_${NAME} target that compiles a Qt5 plugin from
# ${NAME}.cc, ${NAME}plugin.cc with plugin dependencies given by ${DEPENDS}.
function(create_plugin_target)
  cmake_parse_arguments(PLUGIN_ARGS "" "NAME" "DEPENDS" ${ARGN})
  
  generate_target_name(target_name ${PLUGIN_ARGS_NAME})
  message(STATUS "Generating Qt5 plugin '${target_name}'")
  
  
  check_file_exists("widgets/${PLUGIN_ARGS_NAME}.cc")
  check_file_exists("${PLUGIN_ARGS_NAME}plugin.cc")
  add_library(${target_name} SHARED
    widgets/${PLUGIN_ARGS_NAME}.cc
    ${PLUGIN_ARGS_NAME}plugin.cc
  )
  
  target_include_directories(${target_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
  
  # I can't figure out a way to specify the order of library loading
  # MetaData does not seem to work.
  # A simple but not 100% safe workaround is to add proper prefix to the
  # library name.
  
  # Level 0: low-level widgets that don't have dependency
  #       1: widgets that depend on level 0 widgets
  if ("${PLUGIN_ARGS_DEPENDS}" STREQUAL "")
    set(LEVEL 0)
  else()
    set(LEVEL 1)
  endif()
  
  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME "qjulia2_${LEVEL}_${PLUGIN_ARGS_NAME}")
  
  foreach(dep_name ${PLUGIN_ARGS_DEPENDS})
    generate_target_name(dep_target_name ${dep_name})
    target_link_libraries(${target_name} ${dep_target_name})
  endforeach()
  
  target_link_libraries(${target_name} Qt5::Widgets)
  
  install(TARGETS ${target_name}
    LIBRARY DESTINATION "${QT_SYSTEM_PLUGIN_DESIGNER_DIR}"
  )

endfunction()

create_plugin_target(NAME controlbar)
create_plugin_target(NAME lightcontrol DEPENDS controlbar)
create_plugin_target(NAME xyzcoords DEPENDS controlbar)
create_plugin_target(NAME sphericalcoords DEPENDS controlbar)
create_plugin_target(NAME colorlch DEPENDS controlbar)
