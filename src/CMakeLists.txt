add_library(optionspricer
  BlackScholes.cpp
  BinomialTree.cpp
  MonteCarlo.cpp
)

target_include_directories(optionspricer PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(optionspricer PRIVATE -Wall -Wextra)

# AVX2 intrinsics in MonteCarlo.cpp are x86-only; only pass -mavx2 -mfma when
# actually targeting x86_64, so native arm64 builds keep using the scalar
# fallback instead of failing to compile.
set(_optionspricer_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
  list(GET CMAKE_OSX_ARCHITECTURES 0 _optionspricer_target_arch)
endif()
if(_optionspricer_target_arch MATCHES "^(x86_64|AMD64|amd64)$")
  set_source_files_properties(MonteCarlo.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
