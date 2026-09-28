# Copyright (c) 2026 The LightGBM developers. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

function(configure_windows_cuda_device_link target)
  if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio")
    # The final target needs a CUDA source for Visual Studio to run device linking.
    target_sources(${target} PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/cuda/cuda_link.cu")
    # CMake lists object-library inputs relative to the build directory. NVIDIA's
    # ResolvePaths task searches AdditionalLibraryDirectories when collecting
    # CudaLink timestamp dependencies, so it must also search this directory.
    target_link_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endfunction()
