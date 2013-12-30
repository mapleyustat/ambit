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

set(TENSOR_SOURCE_FILES
    dense_tensor.cc
    indices.cc
    tensor_mult_dense.cc
    tensor_print_dense.cc
    tensor_scale_dense.cc
    tensor_size_dense.cc
    tensor_slice_dense.cc
    tensor_sum_dense.cc
    util.cc
)

set(TENSOR_HEADER_FILES
    composite_tensor.h
    dense_tensor.h
    local_tensor.h
    indices.h
    indexable_tensor.h
    tensor.h
    util.h
)

if (MPI_CXX_FOUND)
    list(APPEND TENSOR_SOURCE_FILES cyclops_tensor.cc)
    list(APPEND TENSOR_HEADER_FILES cyclops_tensor.h)
endif()

add_library(tensor ${TENSOR_SOURCE_FILES} ${TENSOR_HEADER_FILES})

