# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Define an OBJECT library with standard include dirs and flags for
# rocjitsu sub-components. ROCJITSU_INCLUDE_DIR and ROCJITSU_SRC_DIR
# must be set before including this module.
#
# Apply the common include paths and warnings to one rocjitsu object library.
function(_rj_configure_object_library name)
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(
        ${name}
        PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR} ${HSA_INCLUDE_DIR}
    )
    target_link_libraries(${name} PRIVATE ${ARGN})
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(
            ${name}
            PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden
        )
    endif()
endfunction()

# Usage: rj_add_object_library(<name> <sources...>)
function(rj_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    _rj_configure_object_library(${name} util simdojo_headers)
endfunction()

# Define one statically linked ISA provider. The provider owns its C++ source,
# self-contained declaration header, and provider function. CMake only packages
# those files with the implementation target.
#
# Usage:
#   rj_add_isa_target_provider(
#       <provider-target>
#       SOURCE <provider-source>
#       HEADER <provider-header>
#       IMPLEMENTATION <target>
#       [BUILTIN]
#   )
function(rj_add_isa_target_provider name)
    set(options BUILTIN)
    set(oneValueArgs SOURCE HEADER IMPLEMENTATION)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})
    foreach(_required SOURCE HEADER IMPLEMENTATION)
        if(NOT ARG_${_required})
            message(
                FATAL_ERROR
                "rj_add_isa_target_provider(${name}) requires ${_required}"
            )
        endif()
    endforeach()
    if(TARGET ${name})
        message(FATAL_ERROR "ISA provider target '${name}' already exists")
    endif()

    set(_registration_target "${name}__registration")
    add_library(${_registration_target} OBJECT "${ARG_SOURCE}")
    _rj_configure_object_library(
        ${_registration_target}
        util
        simdojo_headers
        ${ARG_IMPLEMENTATION}
    )

    add_library(${name} INTERFACE)
    target_sources(${name} INTERFACE $<TARGET_OBJECTS:${_registration_target}>)
    get_target_property(_implementation_type ${ARG_IMPLEMENTATION} TYPE)
    if(_implementation_type STREQUAL "OBJECT_LIBRARY")
        target_sources(
            ${name}
            INTERFACE $<TARGET_OBJECTS:${ARG_IMPLEMENTATION}>
        )
    endif()
    target_link_libraries(${name} INTERFACE ${ARG_IMPLEMENTATION})
    set_target_properties(
        ${name}
        PROPERTIES
            RJ_ISA_PROVIDER_HEADER "${ARG_HEADER}"
            RJ_ISA_PROVIDER_IMPLEMENTATION "${ARG_IMPLEMENTATION}"
    )
    if(ARG_BUILTIN)
        set_property(GLOBAL APPEND PROPERTY RJ_BUILTIN_ISA_PROVIDERS ${name})
    endif()
endfunction()

# Assemble one consumer-owned immutable registry from an exact provider list.
# The generated target header contains only includes for the selected providers.
# Checked-in C++ owns registry construction and public enum lookup.
function(rj_add_isa_target_registry name)
    set(options DEFAULT)
    set(oneValueArgs ACCESSOR)
    set(multiValueArgs PROVIDERS)
    cmake_parse_arguments(
        ARG
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    if(NOT ARG_PROVIDERS)
        message(
            FATAL_ERROR
            "rj_add_isa_target_registry(${name}) requires PROVIDERS"
        )
    endif()
    if(NOT ARG_ACCESSOR)
        string(MAKE_C_IDENTIFIER "rj_get_${name}" ARG_ACCESSOR)
    endif()

    string(MAKE_C_IDENTIFIER "${name}" _header_guard_base)
    string(TOUPPER "${_header_guard_base}" _header_guard_base)
    set(_header_guard "${_header_guard_base}_H_")
    set(_header_name "${name}.h")
    set(_header "${CMAKE_CURRENT_BINARY_DIR}/${_header_name}")
    string(
        CONCAT _header_content
        "// Generated consumer-specific static ISA registry declaration.\n"
        "#ifndef ${_header_guard}\n"
        "#define ${_header_guard}\n\n"
        "#include \"rocjitsu/isa/target_registry.h\"\n\n"
        "namespace rocjitsu {\n"
        "const IsaTargetRegistry &${ARG_ACCESSOR}();\n"
        "} // namespace rocjitsu\n\n"
        "#endif // ${_header_guard}\n"
    )

    set(_target_headers_content)
    set(_provider_implementations)
    foreach(_provider IN LISTS ARG_PROVIDERS)
        if(NOT TARGET ${_provider})
            message(FATAL_ERROR "unknown ISA provider target '${_provider}'")
        endif()
        get_target_property(
            _provider_header
            ${_provider}
            RJ_ISA_PROVIDER_HEADER
        )
        get_target_property(
            _provider_implementation
            ${_provider}
            RJ_ISA_PROVIDER_IMPLEMENTATION
        )
        if(NOT _provider_header OR NOT _provider_implementation)
            message(FATAL_ERROR "target '${_provider}' is not an ISA provider")
        endif()
        string(
            APPEND _target_headers_content
            "#include \"${_provider_header}\"\n"
        )
        list(APPEND _provider_implementations ${_provider_implementation})
    endforeach()

    set(_target_headers_name "${name}_targets.h")
    set(_target_headers "${CMAKE_CURRENT_BINARY_DIR}/${_target_headers_name}")
    file(GENERATE OUTPUT "${_header}" CONTENT "${_header_content}")
    file(
        GENERATE OUTPUT "${_target_headers}"
        CONTENT "${_target_headers_content}"
    )

    set(_registration_target "${name}__composition")
    add_library(
        ${_registration_target}
        OBJECT
        "${ROCJITSU_SRC_DIR}/rocjitsu/isa/target_registry_composition.cpp"
    )
    _rj_configure_object_library(
        ${_registration_target}
        util
        simdojo_headers
        ${_provider_implementations}
    )
    target_include_directories(
        ${_registration_target}
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
    )
    target_compile_definitions(
        ${_registration_target}
        PRIVATE
            "RJ_ISA_TARGET_HEADERS=\"${_target_headers_name}\""
            "RJ_ISA_TARGET_REGISTRY_ACCESSOR=${ARG_ACCESSOR}"
    )
    if(ARG_DEFAULT)
        target_compile_definitions(
            ${_registration_target}
            PRIVATE RJ_ISA_TARGET_REGISTRY_DEFAULT
        )
    endif()
    add_library(${name} INTERFACE)
    target_sources(${name} INTERFACE $<TARGET_OBJECTS:${_registration_target}>)
    target_link_libraries(${name} INTERFACE ${ARG_PROVIDERS})
    target_include_directories(${name} INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")
    set_target_properties(
        ${name}
        PROPERTIES
            RJ_ISA_REGISTRY_ACCESSOR "${ARG_ACCESSOR}"
            RJ_ISA_REGISTRY_HEADER "${_header_name}"
    )
endfunction()
