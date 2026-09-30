// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Prints quant_row_layout.h's offsets so iron/tests/quant_planar.py can diff them against the
// packer's. The two sides of this byte contract have no common type system; K021 is what that
// costs (ppl 4.25e9 on device, through a compile, a link, a numpy round-trip and 5/5 identical
// runs), and a numpy test cannot see it because it re-derives values from the packed bytes.
//
//   clang++ -std=c++17 -DPLANAR=<0|1> -I aie_kernels/generic \
//       iron/tests/quant_row_layout_host.cc -o /tmp/qrl && /tmp/qrl <K> <group> <bits> <rows>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "quant_row_layout.h"

namespace {

// A runtime mirror of quant_row_at<>, which the kernel instantiates on compile-time constants.
// main() checks the two agree before trusting either.
quant_row_offsets at(uint32_t row, uint32_t row_stride, uint32_t header, uint32_t payload) {
#if PLANAR
  const uint32_t base = (row / ROW_GROUP) * (ROW_GROUP * row_stride);
  const uint32_t i = row % ROW_GROUP;
  return {base + ROW_GROUP * payload + i * header, base + i * payload};
#else
  const uint32_t rowp = row * row_stride;
  return {rowp, rowp + header};
#endif
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <K> <group_size> <bits:4|8> <rows>\n", argv[0]);
    return 2;
  }
  const uint32_t K = std::atoi(argv[1]);
  const uint32_t g = std::atoi(argv[2]);
  const uint32_t bits = std::atoi(argv[3]);
  const uint32_t rows = std::atoi(argv[4]);
  const uint32_t payload = (bits == 4) ? K / 2 : K;
  const uint32_t header = 4 * (K / g);  // f32 scale per group; the affine header is 4 B too
  const uint32_t stride = header + payload;

  // Row 0 of the second block: the one place a wrong block base shows.
  const quant_row_offsets t = quant_row_at<1156, 132, 1024>(ROW_GROUP);
  const quant_row_offsets r = at(ROW_GROUP, 1156, 132, 1024);
  if (t.header != r.header || t.payload != r.payload) {
    std::fprintf(stderr, "template and runtime forms disagree\n");
    return 1;
  }

  for (uint32_t row = 0; row < rows; row++) {
    const quant_row_offsets o = at(row, stride, header, payload);
    std::printf("%u %u %u\n", row, o.header, o.payload);
  }
  return 0;
}
