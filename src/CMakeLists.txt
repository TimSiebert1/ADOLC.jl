cmake_minimum_required(VERSION 3.20.0)

project(adolc_wrap)

set(CMAKE_CXX_STANDARD 17)

find_package(JlCxx REQUIRED)
find_library(ADOLC_LIB adolc REQUIRED PATHS "${ADOLC_DIR}/lib64")

set(CMAKE_INSTALL_RPATH "${JlCxx_DIR}/../")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH 1)


add_library(${PROJECT_NAME} SHARED adolc_interface.cpp)
#add_executable(run_test.exe adolc_interface.cpp)
target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC "${ADOLC_DIR}/include/"
    PUBLIC "${ADOLC_DIR}/include/adolc"
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE JlCxx::cxxwrap_julia
    PRIVATE ${ADOLC_LIB}
)
