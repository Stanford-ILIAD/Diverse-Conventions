set(SIMULATOR_SRCS
  sim.hpp sim.cpp
)

add_library(hanabi_example_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(hanabi_example_cpu_impl 
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
)

add_library(hanabi_example_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(hanabi_example_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    hanabi_example_cpu_impl
    madrona_mw_cpu
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            hanabi_example_gpu_srcs
        SOURCES_DEFN
            HANABI_SRC_LIST
        FLAGS_DEFN
            HANABI_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(hanabi_example_mgr PRIVATE
    madrona_mw_gpu
    hanabi_example_gpu_srcs
    )
endif ()

target_compile_definitions(hanabi_example_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_hanabi_example_python
    bindings.cpp
)

target_link_libraries(madrona_hanabi_example_python PRIVATE
    madrona_hdrs
    hanabi_example_mgr
)
