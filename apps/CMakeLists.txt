add_executable(options_pricer main.cpp)
target_link_libraries(options_pricer PRIVATE optionspricer)
target_compile_options(options_pricer PRIVATE -Wall -Wextra)
