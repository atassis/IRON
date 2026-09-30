// SPDX-License-Identifier: Apache-2.0
// Fused flash attention for batched prefill: the QK, softmax and x V workers of one pipeline, on
// mm.cc's 8x8 tiles. QK computes S^T = K . Q^T so no operand is read column-major; the softmax
// works on S^T, where a tile lane is (key, row), so each row's running state is one vector lane.

#include "softmax.cc"
#include "mm.cc"

#ifndef FA_ROWS
#define FA_ROWS 16
#endif

namespace fa {
constexpr int T = 8;
constexpr int ROWS = FA_ROWS;
constexpr int KEYS = 64;
constexpr int NZ = KEYS / T;
constexpr int NJ = ROWS / T;
constexpr float LOG2E = 1.4453125f;
constexpr float LOWEST = -3.3895313892515355e38f;   // the lowest finite bf16, 0xFF7F

alignas(64) static const int32_t kKeyInTile[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};

static inline aie::vector<bfloat16, T> fold_max(aie::vector<bfloat16, 64> v)
{
    aie::vector<bfloat16, 32> a = aie::max(v.extract<32>(0), v.extract<32>(1));
    aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
    return aie::max(b.extract<8>(0), b.extract<8>(1));
}

static inline aie::vector<float, T> fold_sum(aie::vector<float, 64> v)
{
    aie::vector<float, 32> a = aie::add(v.extract<32>(0), v.extract<32>(1));
    aie::vector<float, 16> b = aie::add(a.extract<16>(0), a.extract<16>(1));
    return aie::add(b.extract<8>(0), b.extract<8>(1));
}

static inline aie::vector<float, ROWS> to_f32(aie::vector<bfloat16, ROWS> v)
{
    aie::accum<accfloat, ROWS> a;
    a.from_vector(v);
    return a.to_vector<float>();
}

// Lane r*8+h of an [8 rows][8 cols] tile gets c[r].
static inline aie::vector<float, 64> row_factors(const float *c)
{
    return aie::concat(aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]),
                       aie::broadcast<float, 8>(c[2]), aie::broadcast<float, 8>(c[3]),
                       aie::broadcast<float, 8>(c[4]), aie::broadcast<float, 8>(c[5]),
                       aie::broadcast<float, 8>(c[6]), aie::broadcast<float, 8>(c[7]));
}

// Lane k*8+r of an S^T tile in row tile j gets widths[j*8 + r].
static inline aie::vector<int32, 64> tile_widths(const int32_t *widths, int j)
{
    aie::vector<int32, 8> w = aie::load_v<8>(widths + j * T);
    return aie::concat(w, w, w, w, w, w, w, w);
}

static inline aie::mask<64> visible(int32_t base, int z, const aie::vector<int32, 64> &w64)
{
    return aie::lt(aie::add(aie::load_v<64>(kKeyInTile), base + z * T), w64);
}

// Keys below a row's lower bound are masked as well (a sliding window), when lo is given.
static inline aie::mask<64> visible(int32_t base, int z, const aie::vector<int32, 64> &w64,
                                    const int32_t *lo, int j)
{
    aie::mask<64> v = visible(base, z, w64);
    if (!lo)
        return v;
    aie::vector<int32, 64> k = aie::add(aie::load_v<64>(kKeyInTile), base + z * T);
    return v & aie::ge(k, tile_widths(lo, j));
}
} // namespace fa

extern "C" {

void fa_zero_sT(float *s)
{
    for (int i = 0; i < fa::KEYS * fa::ROWS; i += 16)
        aie::store_v(s + i, aie::zeros<float, 16>());
}

void fa_zero_o(float *o, int32_t n)
{
    for (int32_t i = 0; i < n; i += 16)
        aie::store_v(o + i, aie::zeros<float, 16>());
}

// One delivered Q slice into slice `slice` of Q^T, scaled by log2(e) so the softmax can use exp2.
void fa_qT_slice(bfloat16 *q, bfloat16 *qT, int32_t slice)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const aie::vector<bfloat16, 64> l2e = aie::broadcast<bfloat16, 64>((bfloat16)fa::LOG2E);
    bfloat16 *dst = qT + slice * fa::ROWS * 64;
    for (int kt = 0; kt < 8; kt++)
        for (int rt = 0; rt < fa::NJ; rt++) {
            aie::vector<bfloat16, 64> v = aie::load_v<64>(q + (rt * 8 + kt) * 64);
            aie::vector<bfloat16, 64> s = aie::mul(v, l2e).to_vector<bfloat16>();
            aie::store_v(dst + (kt * fa::NJ + rt) * 64, aie::transpose(s, fa::T, fa::T));
        }
}

void fa_qkT_slice(bfloat16 *k, bfloat16 *qT, float *sT, int32_t slice)
{
    matmul_bf16_f32(k, qT + slice * DIM_K * DIM_N, sT);
}

// Softmax state: m[ROWS], l[ROWS] as f32, then int32 {block, min width, max width, max lower
// bound}.
void fa_sm_round_init(float *st, int32_t *widths)
{
    aie::store_v(st, aie::broadcast<float, fa::ROWS>(-INFINITY));
    aie::store_v(st + fa::ROWS, aie::zeros<float, fa::ROWS>());
    int32_t lo = widths[0], hi = widths[0];
    for (int r = 1; r < fa::ROWS; r++) {
        lo = widths[r] < lo ? widths[r] : lo;
        hi = widths[r] > hi ? widths[r] : hi;
    }
    int32_t *meta = (int32_t *)(st + 2 * fa::ROWS);
    meta[0] = 0;
    meta[1] = lo;
    meta[2] = hi;
    meta[3] = 0;
}

// widths = hi[ROWS] then lo[ROWS]: row r sees keys lo[r] <= key < hi[r].
void fa_sm_round_init_lo(float *st, int32_t *widths)
{
    fa_sm_round_init(st, widths);
    int32_t mlo = widths[fa::ROWS];
    for (int r = 1; r < fa::ROWS; r++)
        mlo = widths[fa::ROWS + r] > mlo ? widths[fa::ROWS + r] : mlo;
    ((int32_t *)(st + 2 * fa::ROWS))[3] = mlo;
}

// One block of online softmax for all rows at once. Keys at or past a row's width are selected to
// -inf (mask != 0). cl gets each row's correction for this block, then the running sum. The first
// block's correction is 1: the accumulator it would scale is still zero.
static void sm_block(float *sT, bfloat16 *p, float *cl, float *st, int32_t *widths, int32_t mask,
                     const int32_t *lo)
{
    using namespace fa;
    ::aie::set_rounding(FLASH_ROUNDING_MODE);
    int32_t *meta = (int32_t *)(st + 2 * ROWS);
    const int32_t blk = meta[0], base = blk * KEYS;
    meta[0] = blk + 1;
    const aie::vector<float, ROWS> l_prev = aie::load_v<ROWS>(st + ROWS);

    if (mask && base >= meta[2]) {
        for (int i = 0; i < ROWS * KEYS; i += 64)
            aie::store_v(p + i, aie::zeros<bfloat16, 64>());
        aie::store_v(cl, aie::broadcast<float, ROWS>(1.0f));
        aie::store_v(cl + ROWS, l_prev);
        return;
    }
    const bool partial = mask && (base + KEYS > meta[1] || base < meta[3]);

    // Per-row-tile pieces go through memory: extract/insert<8> with a loop index on a 32-lane
    // vector miscompiles (task peano-extract-subvector-runtime-index).
    alignas(64) bfloat16 bm[ROWS];
    for (int j = 0; j < NJ; j++) {
        const aie::vector<int32, 64> w64 = tile_widths(widths, j);
        aie::vector<bfloat16, 64> mx = aie::broadcast<bfloat16, 64>((bfloat16)-INFINITY);
        for (int z = 0; z < NZ; z++) {
            aie::vector<float, 64> s = aie::load_v<64>(sT + (z * NJ + j) * 64);
            if (partial)
                s = aie::select((float)-INFINITY, s, visible(base, z, w64, lo, j));
            aie::accum<accfloat, 64> a;
            a.from_vector(s);
            mx = aie::max(mx, a.to_vector<bfloat16>());
        }
        aie::store_v(bm + j * T, fold_max(mx));
    }
    aie::vector<bfloat16, ROWS> bmax = aie::load_v<ROWS>(bm);
    // With a lower bound a row can start with fully masked blocks; its max stays at the lowest
    // finite bf16 instead of -inf, so the next correction is exp2(finite), not exp2(-inf + inf).
    if (lo)
        bmax = aie::max(bmax, aie::broadcast<bfloat16, ROWS>(fa::LOWEST));

    const aie::vector<float, ROWS> m_prev = aie::load_v<ROWS>(st);
    aie::vector<bfloat16, ROWS> m_new = bmax;
    aie::vector<float, ROWS> corr = aie::broadcast<float, ROWS>(1.0f);
    if (blk > 0) {
        aie::accum<accfloat, ROWS> mp;
        mp.from_vector(m_prev);
        m_new = aie::max(mp.to_vector<bfloat16>(), bmax);
        aie::accum<accfloat, ROWS> d = aie::sub(mp, m_new);
        corr = to_f32(aie::exp2<bfloat16>(d.to_vector<float>()));
    }

    alignas(64) bfloat16 mn[ROWS];
    alignas(64) float bs[ROWS];
    aie::store_v(mn, m_new);
    for (int j = 0; j < NJ; j++) {
        const aie::vector<int32, 64> w64 = tile_widths(widths, j);
        const aie::vector<bfloat16, T> m8 = aie::load_v<T>(mn + j * T);
        const aie::vector<bfloat16, 64> mt = aie::concat(m8, m8, m8, m8, m8, m8, m8, m8);
        aie::accum<accfloat, 64> sum = aie::zeros<accfloat, 64>();
        for (int z = 0; z < NZ; z++) {
            aie::accum<accfloat, 64> a;
            a.from_vector(aie::load_v<64>(sT + (z * NJ + j) * 64));
            a = aie::sub(a, mt);
            aie::vector<bfloat16, 64> e = aie::exp2<bfloat16>(a.to_vector<float>());
            if (partial)
                e = aie::select((bfloat16)0.0f, e, visible(base, z, w64, lo, j));
            sum = aie::add(sum, e);
            aie::store_v(p + (j * NZ + z) * 64, aie::transpose(e, T, T));
        }
        aie::store_v(bs + j * T, fold_sum(sum.to_vector<float>()));
    }
    const aie::vector<float, ROWS> bsum = aie::load_v<ROWS>(bs);

    aie::vector<float, ROWS> l = bsum;
    if (blk > 0)
        l = aie::add(aie::mul(l_prev, corr).to_vector<float>(), bsum);
    aie::store_v(st, to_f32(m_new));
    aie::store_v(st + ROWS, l);
    aie::store_v(cl, corr);
    aie::store_v(cl + ROWS, l);
}

void fa_smT_block(float *sT, bfloat16 *p, float *cl, float *st, int32_t *widths, int32_t mask)
{
    sm_block(sT, p, cl, st, widths, mask, nullptr);
}

static void copy_x2(bfloat16 *p0, bfloat16 *p1, float *cl0, float *cl1)
{
    for (int i = 0; i < fa::ROWS * fa::KEYS; i += 64)
        aie::store_v(p1 + i, aie::load_v<64>(p0 + i));
    for (int i = 0; i < 2 * fa::ROWS; i += 16)
        aie::store_v(cl1 + i, aie::load_v<16>(cl0 + i));
}

// The softmax for two x V workers with a per-row lower bound (widths as fa_sm_round_init_lo).
void fa_smT_block_x2_lo(float *sT, bfloat16 *p0, bfloat16 *p1, float *cl0, float *cl1, float *st,
                        int32_t *widths, int32_t mask)
{
    sm_block(sT, p0, cl0, st, widths, mask, widths + fa::ROWS);
    copy_x2(p0, p1, cl0, cl1);
}

// The softmax for two x V workers: each gets its own P and cl, since a FIFO read by both would go
// by DMA (K031).
void fa_smT_block_x2(float *sT, bfloat16 *p0, bfloat16 *p1, float *cl0, float *cl1, float *st,
                     int32_t *widths, int32_t mask)
{
    sm_block(sT, p0, cl0, st, widths, mask, nullptr);
    copy_x2(p0, p1, cl0, cl1);
}

void fa_pv_slice(bfloat16 *p, bfloat16 *v, float *o, int32_t slice)
{
    matmul_bf16_f32(p, v, o + slice * DIM_M * DIM_N);
}

// Rescale every row tile of O that holds a correction other than exactly 1 (compared as bits: a
// scalar f32 compare is a soft-float call here).
void fa_o_rescale(float *o, float *cl, int32_t n_slices)
{
    for (int rt = 0; rt < fa::NJ; rt++) {
        uint32_t bits[fa::T];
        __builtin_memcpy(bits, cl + rt * fa::T, sizeof bits);
        bool unit = true;
        for (int r = 0; r < fa::T; r++)
            unit = unit && bits[r] == 0x3f800000u;
        if (unit)
            continue;
        const aie::vector<float, 64> f = fa::row_factors(cl + rt * fa::T);
        for (int32_t sl = 0; sl < n_slices; sl++)
            for (int nt = 0; nt < 8; nt++) {
                float *t = o + sl * fa::ROWS * 64 + (rt * 8 + nt) * 64;
                aie::store_v(t, aie::mul(aie::load_v<64>(t), f).to_vector<float>());
            }
    }
}

void fa_inv_l(float *cl, float *inv)
{
    for (int r = 0; r < fa::ROWS; r++)
        inv[r] = 1.0f / cl[fa::ROWS + r];
}

// One slice of the normalised output, [16 rows][64] bf16 row-major.
void fa_o_finish_slice(float *o, float *inv, bfloat16 *out, int32_t slice)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    for (int rt = 0; rt < fa::NJ; rt++) {
        const aie::vector<float, 64> f = fa::row_factors(inv + rt * fa::T);
        for (int nt = 0; nt < 8; nt++) {
            const float *t = o + slice * fa::ROWS * 64 + (rt * 8 + nt) * 64;
            aie::vector<bfloat16, 64> b = aie::mul(aie::load_v<64>(t), f).to_vector<bfloat16>();
            for (int r = 0; r < fa::T; r++)
                aie::store_v(out + (rt * 8 + r) * 64 + nt * 8, b.extract<8>(r));
        }
    }
}
}
