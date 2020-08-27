cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(superglue)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(
        "-g"
        "-Wall"
        "-Wextra"
        "-pedantic"
        "-Wstrict-aliasing"
        "-pedantic"
        "-Wunreachable-code"
        "-Wcast-align"
        "-Wcast-qual"
        "-Wctor-dtor-privacy"
        "-Wdisabled-optimization"
        "-Wformat=2"
        "-Winit-self"
        "-Wmissing-include-dirs"
        "-Wold-style-cast"
        "-Woverloaded-virtual"
        "-Wredundant-decls"
        "-Wshadow"
        "-Wsign-promo"
        "-Wstrict-overflow"
        "-Wswitch-default"
        "-Wundef"
        "$<$<CONFIG:RELEASE>:-O3>")
endif()

###########################################################################
#                          !! DEBUG DELETE LATER!!                        #
###########################################################################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   add_compile_options (-fcolor-diagnostics)
endif ()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


add_library(features superPoint.cpp)
target_link_libraries(features ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(features PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(superglue main.cpp)
target_link_libraries(superglue ${TORCH_LIBRARIES} ${OpenCV_LIBS} features)
target_include_directories(superglue PUBLIC ${OpenCV_INCLUDE_DIRS} include)


file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/SuperPoint.zip
  ${CMAKE_CURRENT_SOURCE_DIR}/SuperGlue.zip
  DESTINATION ${CMAKE_BINARY_DIR}
)
