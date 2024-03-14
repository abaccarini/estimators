cmake_minimum_required(VERSION 3.16)

project(estimator LANGUAGES C CXX)


SET(CMAKE_COLOR_MAKEFILE ON)

set(CMAKE_C_FLAGS "-Wall")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-Wall -std=c++2a")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(GMP REQUIRED)
find_package(GSL REQUIRED)
find_package(MPFR REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
# find_package(nlohmann_json REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

INCLUDE_DIRECTORIES(
  "${MPFR_INCLUDES}"
  "${GMP_INCLUDES}"
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/src/headers"
  "${EIGEN3_INCLUDE_DIR}"
  "${Boost_INCLUDE_DIRS}"
)

# set(SOURCE_FILES src/main.c)
file(GLOB SOURCES
  "src/*.cpp"
)

add_executable(estimator ${SOURCES})

target_link_libraries(estimator ${MPFR_LIBRARIES} ${GMP_LIBRARIES} ${GSL_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS estimator
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)

add_custom_target(clean-all
  make clean && rm -fR ${CMAKE_BINARY_DIR}/CMakeCache.txt
  ${CMAKE_BINARY_DIR}/cmake_install.cmake
  ${CMAKE_BINARY_DIR}/Makefile
  ${CMAKE_BINARY_DIR}/CMakeFiles
  && rmdir ${CMAKE_BINARY_DIR}/build
)