add_executable(floating_point_predicates_precision floating_point_predicates_precision.cu)
target_include_directories(floating_point_predicates_precision PRIVATE ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../lib/)

add_executable(moving_device_mesh moving_device_mesh.cu)
target_link_libraries(moving_device_mesh PRIVATE ${CUDA_LIBRARIES})
target_include_directories(moving_device_mesh PRIVATE ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../lib/ ../include)

add_executable(diffusion_limited_aggregation diffusion_limited_aggregation.cu)
target_link_libraries(diffusion_limited_aggregation PRIVATE ${CUDA_LIBRARIES})
target_include_directories(diffusion_limited_aggregation PRIVATE ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../lib/ ../include)

add_executable(moving_periodic_device_mesh moving_periodic_device_mesh.cu)
target_link_libraries(moving_periodic_device_mesh PRIVATE ${CGAL_LIBRARIES})
target_include_directories(moving_periodic_device_mesh PRIVATE ../lib/glm ../lib/gcem/include ../lib/libmorton/include ../lib/ ../include)
