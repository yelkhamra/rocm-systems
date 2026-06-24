# HRR In-Tree Capture Layer
# Adds capture source files to the amdhip64 target.
# No separate library — capture is compiled into amdhip64 directly.

target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_generated.cpp
)

# hrr_api_args.h and all capture headers live here.
target_include_directories(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# shim/sitecustomize.py is a runtime-only artifact (the caching-allocator
# sub-allocation map capture shim). It is not compiled or linked; it is loaded
# by the captured Python workload via PYTHONPATH at run time. See DESIGN.md
# ("Caching-Allocator Sub-Allocation Map") and hrr_suballoc.h.

# Playback tools (hrr-playback, hrr_reader/hrr_playback libs)
add_subdirectory(playback)
