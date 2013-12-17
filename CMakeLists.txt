#
#  Copyright (C) 2013  Justin Turney
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

project(ambit)
cmake_minimum_required(VERSION 2.8)

get_filename_component(ROOT_SRC_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
set(CMAKE_MODULE_PATH "${ROOT_SRC_DIR}/cmake/")

# CTF stuff
if (DEFINED CTF_DIR)
    message("CTF_DIR = ${CTF_DIR}")
else()
    message(FATAL_ERROR "CTF_DIR not defined. Unable to continue.")
endif()

set(CTF_INCLUDE_DIRS ${CTF_DIR}/include)
set(CTF_LIBRARIES ${CTF_DIR}/lib/libctf.a)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(LAPACK REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost 1.49 COMPONENTS python REQUIRED)

include_directories(
    ${CTF_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_subdirectory(src)

