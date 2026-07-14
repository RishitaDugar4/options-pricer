include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(optionspricer_tests
  test_black_scholes.cpp
  test_binomial_tree.cpp
  test_monte_carlo.cpp
)

target_link_libraries(optionspricer_tests PRIVATE optionspricer GTest::gtest_main)
target_compile_options(optionspricer_tests PRIVATE -Wall -Wextra)

include(GoogleTest)
gtest_discover_tests(optionspricer_tests)
