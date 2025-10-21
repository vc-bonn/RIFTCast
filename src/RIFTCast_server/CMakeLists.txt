# Collect all source files in this directory
file(GLOB_RECURSE source "**.h*" "**.c*")
# Retrieve target name from current folder name
get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add target executable
ATCG_add_executable(${TARGET_NAME} target_name "${source}")

if(TARGET ${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} PUBLIC RIFTCast_lib)
endif()
