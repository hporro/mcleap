add_executable(test_frnn test_frnn.cu)
target_include_directories(test_frnn PRIVATE ../lib/tinytest ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../include)