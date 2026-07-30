/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef QUANT_DEF
#define QUANT_DEF

#define _MAX_VAL   1023.0f
#define _BITS_X    7
#define _BITS_Y    7
#define _BITS_EXP  4
#define _DEPTH_MAX 1.1f

FfxFloat32 sym_ceil(FfxFloat32 x)
{
    return sign(x) * ceil(abs(x));
}

// Pack flow (vec2) and depth -> 32-bit int code
// Layout (MSB->LSB): depth | exponent | signX | signY | mantissaX | mantissaY
// -----------------------------------------------------------------------------
FfxUInt32 qFloat(FfxFloat32x2 flow, FfxFloat32 depth)
{
    FfxInt32   full_bits = _BITS_X + _BITS_Y;
    FfxInt32   max_int   = (1 << full_bits) - 1;
    FfxFloat32 scale     = FfxFloat32(max_int) / _MAX_VAL;

    FfxInt32   depth_bits = 31 - full_bits - 2 - _BITS_EXP;
    FfxInt32   max_int_d  = (1 << depth_bits) - 1;
    FfxFloat32 depth_max  = FfxFloat32(max_int_d) / _DEPTH_MAX;

    FfxInt32 ix = FfxInt32(clamp(sym_ceil(flow.x * scale), -FfxFloat32(max_int), FfxFloat32(max_int)));
    FfxInt32 iy = FfxInt32(clamp(sym_ceil(flow.y * scale), -FfxFloat32(max_int), FfxFloat32(max_int)));
    FfxInt32 id = FfxInt32(round(depth * depth_max));

    FfxInt32 absmax = max(abs(ix), abs(iy));
    FfxInt32 exp    = clamp(findMSB(absmax), 0, (1 << _BITS_EXP) - 1);

    // exp needs to be cast to FfxInt32 here to avoid underflow
    FfxInt32 shift = max(exp - (FfxInt32(_BITS_X) - 1), 0);
    FfxInt32 mx    = clamp((abs(ix) >> shift), 0, (1 << _BITS_X) - 1);
    FfxInt32 my    = clamp((abs(iy) >> shift), 0, (1 << _BITS_Y) - 1);

    FfxInt32 sY_shift    = _BITS_X + _BITS_Y;
    FfxInt32 sX_shift    = sY_shift + 1;
    FfxInt32 exp_shift   = sX_shift + 1;
    FfxInt32 depth_shift = exp_shift + _BITS_EXP;

    FfxUInt32 code = 0;
    code |= (id & max_int_d) << depth_shift;
    code |= (exp & ((1 << _BITS_EXP) - 1)) << exp_shift;
    code |= ((ix < 0) ? 1 : 0) << sX_shift;
    code |= ((iy < 0) ? 1 : 0) << sY_shift;
    code |= (mx & ((1 << _BITS_X) - 1)) << _BITS_Y;
    code |= (my & ((1 << _BITS_Y) - 1));

    code &= 0x7FFFFFFF;

    return code;
}

FfxUInt32 qFloat(FfxFloat32x2 flow, FfxInt32 depth)
{
    FfxInt32   full_bits = _BITS_X + _BITS_Y;
    FfxInt32   max_int   = (1 << full_bits) - 1;
    FfxFloat32 scale     = FfxFloat32(max_int) / _MAX_VAL;

    FfxInt32   depth_bits = 31 - full_bits - 2 - _BITS_EXP;
    FfxInt32   max_int_d  = (1 << depth_bits) - 1;
    FfxFloat32 depth_max  = FfxFloat32(max_int_d) / _DEPTH_MAX;

    FfxInt32 ix = FfxInt32(clamp(sym_ceil(flow.x * scale), -FfxFloat32(max_int), FfxFloat32(max_int)));
    FfxInt32 iy = FfxInt32(clamp(sym_ceil(flow.y * scale), -FfxFloat32(max_int), FfxFloat32(max_int)));
    FfxInt32 id = depth;

    FfxInt32 absmax = max(abs(ix), abs(iy));
    FfxInt32 exp    = clamp(findMSB(absmax), 0, (1 << _BITS_EXP) - 1);

    // exp needs to be cast to FfxInt32 here to avoid underflow
    FfxInt32 shift = max(exp - (FfxInt32(_BITS_X) - 1), 0);
    FfxInt32 mx    = clamp((abs(ix) >> shift), 0, (1 << _BITS_X) - 1);
    FfxInt32 my    = clamp((abs(iy) >> shift), 0, (1 << _BITS_Y) - 1);

    FfxInt32 sY_shift    = _BITS_X + _BITS_Y;
    FfxInt32 sX_shift    = sY_shift + 1;
    FfxInt32 exp_shift   = sX_shift + 1;
    FfxInt32 depth_shift = exp_shift + _BITS_EXP;

    FfxUInt32 code = 0;
    code |= (id & max_int_d) << depth_shift;
    code |= (exp & ((1 << _BITS_EXP) - 1)) << exp_shift;
    code |= ((ix < 0) ? 1 : 0) << sX_shift;
    code |= ((iy < 0) ? 1 : 0) << sY_shift;
    code |= (mx & ((1 << _BITS_X) - 1)) << _BITS_Y;
    code |= (my & ((1 << _BITS_Y) - 1));

    code &= 0x7FFFFFFF;

    return code;
}

FfxFloat32x2 dqFloatMv(FfxUInt32 c)
{
    c &= 0x7FFFFFFF;
    FfxUInt32 sY_shift  = _BITS_X + _BITS_Y;
    FfxUInt32 sX_shift  = sY_shift + 1;
    FfxUInt32 exp_shift = sX_shift + 1;

    FfxInt32 sx = 1 - 2 * FfxInt32((c >> sX_shift) & 1);
    FfxInt32 sy = 1 - 2 * FfxInt32((c >> sY_shift) & 1);

    FfxUInt32 exp = (c >> exp_shift) & ((1 << _BITS_EXP) - 1);
    FfxUInt32 mx  = (c >> _BITS_Y) & ((1 << _BITS_X) - 1);
    FfxUInt32 my  = c & ((1 << _BITS_Y) - 1);

    // exp needs to be cast to FfxInt32 here to avoid underflow
    FfxUInt32 shift = FfxUInt32(max(FfxInt32(exp) - (FfxInt32(_BITS_X) - 1), 0));
    FfxInt32  ix    = sx * FfxInt32(mx << shift);
    FfxInt32  iy    = sy * FfxInt32(my << shift);

    FfxFloat32 inv_scale = _MAX_VAL / FfxFloat32((1 << (_BITS_X + _BITS_Y)) - 1);
    return FfxFloat32x2(FfxFloat32(ix), FfxFloat32(iy)) * inv_scale;
}

#endif
