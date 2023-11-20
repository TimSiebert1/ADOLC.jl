# Use this with "cmake -S. -Bbuild" inside the "find_julia" folder
# to find julia

cmake_minimum_required(VERSION 3.5.0)
project(FindJulia)

set(CMAKE_MODULE_PATH "./")

# this path must contain the "bin" and "include" dirs 
# on MacOS it may be something like 
# /Applications/Julia-1.9.app/Contents/Resources/julia/
#list(APPEND Julia_PREFIX "PATH/TO/JULIA/")

# e.g. /usr/local/bin/julia
#set(APPEND Julia_EXECUTABLE "PATH/TO/BIN/JULIA")

find_package(Julia REQUIRED)