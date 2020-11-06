cmake_minimum_required(VERSION 3.10)

project(WaveFD)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

add_library(prop2DAcoIsoDenQ_DEO2_FDTD prop2DAcoIsoDenQ_DEO2_FDTD.cc)
add_library(prop2DAcoVTIDenQ_DEO2_FDTD prop2DAcoVTIDenQ_DEO2_FDTD.cc)
add_library(prop2DAcoTTIDenQ_DEO2_FDTD prop2DAcoTTIDenQ_DEO2_FDTD.cc)
add_library(prop3DAcoIsoDenQ_DEO2_FDTD prop3DAcoIsoDenQ_DEO2_FDTD.cc)
add_library(prop3DAcoVTIDenQ_DEO2_FDTD prop3DAcoVTIDenQ_DEO2_FDTD.cc)
add_library(prop3DAcoTTIDenQ_DEO2_FDTD prop3DAcoTTIDenQ_DEO2_FDTD.cc)
add_library(illumination illumination.c)
add_library(spacetime spacetime.cc)

find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
target_link_libraries(prop2DAcoIsoDenQ_DEO2_FDTD PUBLIC PkgConfig::FFTW3F)
target_link_libraries(prop2DAcoVTIDenQ_DEO2_FDTD PUBLIC PkgConfig::FFTW3F)
target_link_libraries(prop2DAcoTTIDenQ_DEO2_FDTD PUBLIC PkgConfig::FFTW3F)
target_link_libraries(prop3DAcoIsoDenQ_DEO2_FDTD PUBLIC PkgConfig::FFTW3F)
target_link_libraries(prop3DAcoVTIDenQ_DEO2_FDTD PUBLIC PkgConfig::FFTW3F)
target_link_libraries(prop3DAcoTTIDenQ_DEO2_FDTD PUBLIC PkgConfig::FFTW3F)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message("OpenMP found: ${OpenMP_CXX_FOUND}")
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(prop2DAcoIsoDenQ_DEO2_FDTD PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(prop2DAcoVTIDenQ_DEO2_FDTD PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(prop2DAcoTTIDenQ_DEO2_FDTD PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(prop3DAcoIsoDenQ_DEO2_FDTD PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(prop3DAcoVTIDenQ_DEO2_FDTD PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(prop3DAcoTTIDenQ_DEO2_FDTD PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(spacetime PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(illumination PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS prop2DAcoIsoDenQ_DEO2_FDTD prop2DAcoVTIDenQ_DEO2_FDTD prop2DAcoTTIDenQ_DEO2_FDTD prop3DAcoIsoDenQ_DEO2_FDTD prop3DAcoVTIDenQ_DEO2_FDTD prop3DAcoTTIDenQ_DEO2_FDTD illumination spacetime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
