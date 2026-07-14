add_library(optionspricer
  BlackScholes.cpp
  BinomialTree.cpp
  MonteCarlo.cpp
)

target_include_directories(optionspricer PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(optionspricer PRIVATE -Wall -Wextra)
