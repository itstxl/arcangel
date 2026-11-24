cmake_minimum_required(VERSION 3.24)
project(arcangel CXX)

# ArcAngel: Inkling 2 Mini inference engine for Intel Arc Battlemage (Xe2) cards, SYCL on Level Zero.
# Configure with the oneAPI compiler (install.sh does this for you):
#   source /opt/intel/oneapi/setvars.sh
#   cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DINK_AOT_TARGETS=bmg_g21

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # inkcore also goes into libinkling.so

# Ahead-of-time compile so there is no kernel JIT at start-up (needs ocloc, from intel-graphics-compiler).
# bmg_g21: Arc B570, B580, Pro B50, Pro B60. bmg_g31: Arc Pro B70. Several: "bmg_g21,bmg_g31".
# INK_AOT=OFF builds portable SPIR-V that is JIT-compiled on first use (slow first start).
option(INK_AOT "Compile kernels ahead of time" ON)
set(INK_AOT_TARGETS "bmg_g21" CACHE STRING "Battlemage chips to compile for (bmg_g21, bmg_g31, comma separated)")

set(INK_SYCL_FLAGS -fsycl)
if(INK_AOT)
  string(REPLACE "," ";" _ink_targets "${INK_AOT_TARGETS}")
  list(TRANSFORM _ink_targets PREPEND "intel_gpu_")
  list(JOIN _ink_targets "," _ink_targets)
  list(APPEND INK_SYCL_FLAGS -fsycl-targets=${_ink_targets})
endif()

add_library(inkcore STATIC
  src/gguf_file.cpp
  src/device.cpp
  src/model.cpp
)
target_include_directories(inkcore PUBLIC src)
target_compile_options(inkcore PUBLIC ${INK_SYCL_FLAGS} -Wall -Wextra)
target_link_options(inkcore PUBLIC ${INK_SYCL_FLAGS})

add_executable(ink-probe tools/probe.cpp)
target_link_libraries(ink-probe PRIVATE inkcore)

target_sources(inkcore PRIVATE src/engine.cpp src/fast.cpp src/prefill.cpp)

add_executable(test-forward tests/test_forward.cpp)
target_link_libraries(test-forward PRIVATE inkcore)

# Developer tests against llama.cpp (needs a llama.cpp build with the Inkling support, from source). The
# tests compare the engine against llama.cpp's CPU code, value for value. Off for normal installs.
option(INK_LLAMA_TESTS "Build test-dequant and ref-dump against a llama.cpp build" OFF)
if(INK_LLAMA_TESTS)
  set(INK_LLAMA_DIR "$ENV{HOME}/.local/src/llama.cpp-inkling" CACHE PATH "llama.cpp tree used as the reference")
  set(INK_LLAMA_LIBDIR "${INK_LLAMA_DIR}/build-vk/bin" CACHE PATH "directory holding its libggml-base.so")
  set(_llama_libs ${INK_LLAMA_LIBDIR}/libllama.so ${INK_LLAMA_LIBDIR}/libggml-base.so ${INK_LLAMA_LIBDIR}/libggml.so)

  add_executable(test-dequant tests/test_dequant.cpp)
  target_include_directories(test-dequant PRIVATE ${INK_LLAMA_DIR}/ggml/include)
  target_link_libraries(test-dequant PRIVATE inkcore ${INK_LLAMA_LIBDIR}/libggml-base.so)
  set_target_properties(test-dequant PROPERTIES BUILD_RPATH ${INK_LLAMA_LIBDIR})

  # llama.cpp on the CPU as the per-layer answer key (plain C++, no SYCL). Must stay CPU-only.
  add_executable(ref-dump tools/ref_dump.cpp)
  target_include_directories(ref-dump PRIVATE ${INK_LLAMA_DIR}/include ${INK_LLAMA_DIR}/ggml/include)
  target_link_libraries(ref-dump PRIVATE ${_llama_libs})
  set_target_properties(ref-dump PROPERTIES BUILD_RPATH ${INK_LLAMA_LIBDIR})

  # rewrites chosen tensors of a GGUF in another format (tests formats of bigger quants on a small card)
  add_executable(gguf-retype tests/gguf_retype.cpp)
  target_include_directories(gguf-retype PRIVATE ${INK_LLAMA_DIR}/include ${INK_LLAMA_DIR}/ggml/include)
  target_link_libraries(gguf-retype PRIVATE ${_llama_libs})
  set_target_properties(gguf-retype PROPERTIES BUILD_RPATH ${INK_LLAMA_LIBDIR})
endif()

add_executable(bench-decode tools/bench_decode.cpp)
target_link_libraries(bench-decode PRIVATE inkcore)

add_executable(bench-prefill tools/bench_prefill.cpp)
target_link_libraries(bench-prefill PRIVATE inkcore)

# C API for the serving layer
add_library(inkling SHARED src/capi.cpp)
target_link_libraries(inkling PRIVATE inkcore)
