# Enumerate all subdirectories that contain a CMakeLists.txt and include them in the cmake project
file(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
foreach(subdircmakelists IN LISTS subdirs)
    get_filename_component(subdir ${subdircmakelists} DIRECTORY)
    add_subdirectory(${subdir})
endforeach()