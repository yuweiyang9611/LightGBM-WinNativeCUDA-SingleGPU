# Copyright (c) 2026 The LightGBM developers. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

# Prepare a build-tree copy of fmt with the upstream CUDA UTF-32 workaround.
# Keeping the change outside the fmt submodule makes the fix reproducible from
# a normal recursive clone without requiring a LightGBM-specific fmt fork.
function(prepare_fmt_for_msvc_cuda output_variable)
  set(_fmt_source_include "${PROJECT_SOURCE_DIR}/external_libs/fmt/include")
  set(_fmt_patched_include "${PROJECT_BINARY_DIR}/fmt_msvc_cuda/include")
  set(_fmt_source_dir "${_fmt_source_include}/fmt")
  set(_fmt_patched_dir "${_fmt_patched_include}/fmt")
  file(MAKE_DIRECTORY "${_fmt_patched_dir}")

  file(GLOB _fmt_headers CONFIGURE_DEPENDS "${_fmt_source_dir}/*.h")
  foreach(_fmt_header IN LISTS _fmt_headers)
    get_filename_component(_fmt_header_name "${_fmt_header}" NAME)
    if(NOT _fmt_header_name STREQUAL "format.h")
      configure_file(
        "${_fmt_header}"
        "${_fmt_patched_dir}/${_fmt_header_name}"
        COPYONLY
      )
    endif()
  endforeach()

  set(_fmt_format_source "${_fmt_source_dir}/format.h")
  set(_fmt_format_patched "${_fmt_patched_dir}/format.h")
  file(READ "${_fmt_format_source}" _fmt_format_content)

  string(CONCAT _fmt_cuda_utf32_original
    [=[  return U"\x9999999a\x828f5c29\x80418938\x80068db9\x8000a7c6\x800010c7"]=]
    "\n"
    [=[         U"\x800001ae\x8000002b"[index];]=]
  )
  string(CONCAT _fmt_cuda_utf32_replacement
    [=[  return uint32_t(u"\x9999\x828f\x8041\x8006\x8000\x8000\x8000\x8000"[index])]=]
    "\n"
    [=[             << 16u |]=]
    "\n"
    [=[         uint32_t(u"\x999a\x5c29\x8938\x8db9\xa7c6\x10c7\x01ae\x002b"[index]);]=]
  )

  string(FIND "${_fmt_format_content}" "${_fmt_cuda_utf32_original}" _fmt_original_position)
  if(NOT _fmt_original_position EQUAL -1)
    string(
      REPLACE
      "${_fmt_cuda_utf32_original}"
      "${_fmt_cuda_utf32_replacement}"
      _fmt_format_content
      "${_fmt_format_content}"
    )
  else()
    string(FIND "${_fmt_format_content}" "${_fmt_cuda_utf32_replacement}" _fmt_replacement_position)
    if(_fmt_replacement_position EQUAL -1)
      message(FATAL_ERROR "The bundled fmt format.h no longer matches the CUDA UTF-32 compatibility patch")
    endif()
  endif()

  set(_fmt_write_patched_header ON)
  if(EXISTS "${_fmt_format_patched}")
    file(READ "${_fmt_format_patched}" _fmt_existing_content)
    if(_fmt_existing_content STREQUAL _fmt_format_content)
      set(_fmt_write_patched_header OFF)
    endif()
  endif()
  if(_fmt_write_patched_header)
    file(WRITE "${_fmt_format_patched}" "${_fmt_format_content}")
  endif()

  set(${output_variable} "${_fmt_patched_include}" PARENT_SCOPE)
endfunction()
