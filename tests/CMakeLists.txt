add_executable(test_frnn test_frnn.cu)
target_include_directories(test_frnn PRIVATE ../lib/tinytest ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../include)

add_executable(test_closest_neighbors test_closest_neighbors.cu)
target_include_directories(test_closest_neighbors PRIVATE ../lib/tinytest ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../include)