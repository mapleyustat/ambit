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

add_definitions("-DROOT_SRC_DIR=\"${ROOT_SRC_DIR}\"")

# MAC: On Mavericks clang++ defaults to std11 being on. So when
# installing Boost via Homebrew BoostPython uses std11 features
# but since I'm using g++-4.8 with MPI it doesn't enable std11
# by default. If you see alot of undefined references in BoostPython
# this could be your issue.
include(CheckCXX11Features)
foreach (flag IN ITEMS
        HAS_CXX11_AUTO
        HAS_CXX11_AUTO_RET_TYPE
        HAS_CXX11_CLASS_OVERRIDE
        HAS_CXX11_CONSTEXPR
        HAS_CXX11_CSTDINT_H
        HAS_CXX11_DECLTYPE
        HAS_CXX11_FUNC
        HAS_CXX11_INITIALIZER_LIST
        HAS_CXX11_LAMBDA
        HAS_CXX11_LIB_REGEX
        HAS_CXX11_LONG_LONG
        HAS_CXX11_NULLPTR
        HAS_CXX11_RVALUE_REFERENCES
        HAS_CXX11_SIZEOF_MEMBER
        HAS_CXX11_STATIC_ASSERT
        HAS_CXX11_VARIADIC_TEMPLATES
        )
    if (${flag})
        add_definitions("-D${flag}")
        message(STATUS "Compiler C++11 support flag ${flag} set")
    endif ()
endforeach (flag)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${CXX11_COMPILER_FLAGS})

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

# Check for restrict keyword
# Builds the macro A_C_RESTRICT form automake
foreach(ac_kw __restrict __restrict__ _Restrict restrict)
  check_cxx_source_compiles(
  "
  typedef int * int_ptr;
  int foo (int_ptr ${ac_kw} ip) {
    return ip[0];
  }
  int main(){
    int s[1];
    int * ${ac_kw} t = s;
    t[0] = 0;
    return foo(t);
  }
  "
  RESTRICT)
  if(RESTRICT)
    set(ac_cv_c_restrict ${ac_kw})
    break()
  endif()
endforeach()
if(RESTRICT)
  add_definitions("-Drestrict=${ac_cv_c_restrict}")
else()
  add_definitions("-Drestrict=")
endif()

include_directories(
    ${CTF_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    src
)

#
# Location of compiled libraries and executables
#
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(examples)

