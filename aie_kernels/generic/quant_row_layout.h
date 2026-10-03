// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Where a packed weight row's header and payload live -- the device half of a byte contract whose
// host half is iron/common/quant.py (`row_offsets`). Free of aie_api on purpose: iron/tests/
// quant_planar.py compiles this with a host compiler and diffs its offsets against the packer's,
// which is the only check that can see the two sides drift.
//
//   PLANAR=0  [n_groups x scale][payload] per row, row r at r*row_stride.
//   PLANAR=1  ROW_GROUP rows as [payload_0..][header_0..], so a payload row starts at a multiple
//             of its own length and the load width can be chosen for speed, not alignment.
#ifndef QUANT_ROW_LAYOUT_H
#define QUANT_ROW_LAYOUT_H

#include <stdint.h>

#ifndef PLANAR
#define PLANAR 0
#endif

// One-row default; designs select a fitting block height. See iron/common/quant.py::derive_row_group.
#ifndef ROW_GROUP
#define ROW_GROUP 1
#endif

struct quant_row_offsets {
  uint32_t header;
  uint32_t payload;
};

// Byte offsets; see iron/common/quant.py for the complete-block tiling contract.
template <uint32_t row_stride, uint32_t header, uint32_t payload>
inline quant_row_offsets quant_row_at(uint32_t row) {
#if PLANAR
  const uint32_t base = (row / ROW_GROUP) * (ROW_GROUP * row_stride);
  const uint32_t i = row % ROW_GROUP;
  return {base + ROW_GROUP * payload + i * header, base + i * payload};
#else
  const uint32_t rowp = row * row_stride;
  return {rowp, rowp + header};
#endif
}

// Runtime-args sibling of quant_row_at, for mv_quant.cc's matvec_int4_dequant_rtk: row_stride/
// header/payload become arguments instead of template constants. No static_assert can check a
// runtime k (K021); the generator guarantees alignment instead (see that function's comment).
inline quant_row_offsets quant_row_at_rt(uint32_t row, uint32_t row_stride, uint32_t header,
                                         uint32_t payload) {
#if PLANAR
  const uint32_t base = (row / ROW_GROUP) * (ROW_GROUP * row_stride);
  const uint32_t i = row % ROW_GROUP;
  return {base + ROW_GROUP * payload + i * header, base + i * payload};
#else
  const uint32_t rowp = row * row_stride;
  return {rowp, rowp + header};
#endif
}

// int4 callers pass r/2: two nibbles per byte halve the load.
#if PLANAR
#define QUANT_ALIGN_OK(load, header, payload, stride) \
  ((payload) % (load) == 0 && (ROW_GROUP * (stride)) % (load) == 0)
#else
#define QUANT_ALIGN_OK(load, header, payload, stride) \
  ((header) % (load) == 0 && (stride) % (load) == 0)
#endif

#endif  // QUANT_ROW_LAYOUT_H
