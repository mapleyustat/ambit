/*
 * Copyright (C) 2013 Devin Matthews
 *
 * This is a slimmed down version of the tensor framework developed by
 * Devin Matthews. The version by Devin was tied to Aquarius. This
 * version is not.
 *
 * Copyright (C) 2013  Justin Turney
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "tensor.h"
#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

namespace ambit {

void first_packed_indices(const int ndim, const int* len, const int* sym, int* idx)
{
    int i;

    if (ndim > 0) idx[0] = 0;
    for (i = 0;i < ndim - 1;i++)
    {
        switch (sym[i])
        {
            case AS:
            case SH:
                idx[i+1] = idx[i] + 1;
                break;
            case SY:
                idx[i+1] = idx[i];
                break;
            case NS:
                idx[i+1] = 0;
                break;
        }
    }
}

bool next_packed_indices(const int ndim, const int* len, const int* sym, int* idx)
{
    int i;

    for (i = 0;i < ndim;i++)
    {
        if (i == ndim - 1)
        {
            if (idx[i] == len[i]-1)
            {
                return false;
            }
            else
            {
                idx[i]++;
                break;
            }
        }
        else
        {
            if ((sym[i] == SY && idx[i] == idx[i+1]) ||
                ((sym[i] == AS || sym[i] == SH) && idx[i] == idx[i+1]-1) ||
                (idx[i] == len[i]-1))
            {
                if (i == 0)
                {
                    idx[i] = 0;
                }
                else if (sym[i-1] == NS)
                {
                    idx[i] = 0;
                }
                else if (sym[i-1] == SY)
                {
                    idx[i] = idx[i-1];
                }
                else // AS and SH
                {
                    idx[i] = idx[i-1] + 1;
                }
            }
            else
            {
                idx[i]++;
                break;
            }
        }
    }

    return (ndim > 0 ? true : false);
}

}
