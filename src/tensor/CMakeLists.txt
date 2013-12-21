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
    cyclops_tensor.cc
    dense_tensor.cc
    util.cc
)

set(TENSOR_HEADER_FILES
    cyclops_tensor.h
    dense_tensor.h
    local_tensor.h
    indexable_tensor.h
    tensor.h
    util.h
)

add_library(tensor ${TENSOR_SOURCE_FILES} ${TENSOR_HEADER_FILES})

