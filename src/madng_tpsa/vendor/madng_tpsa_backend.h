/*
 * Compact MAD-NG-compatible real TPSA C backend for madng-tpsa.
 *
 * The public names and signatures follow MAD-NG's real TPSA C headers so the
 * same Python CFFI layer can target this bundled backend or an external MAD-NG
 * shared library.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef MADNG_TPSA_BACKEND_H
#define MADNG_TPSA_BACKEND_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef _Bool    log_t;
typedef int32_t  idx_t;
typedef int32_t  ssz_t;
typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef double   num_t;
typedef const char *str_t;
typedef unsigned char ord_t;

typedef struct desc_ desc_t;
typedef struct tpsa_ tpsa_t;

extern const ord_t mad_tpsa_dflt;
extern const ord_t mad_tpsa_same;
extern num_t mad_tpsa_eps;

const desc_t* mad_desc_newv(int nv, ord_t mo);
const desc_t* mad_desc_newvp(int nv, ord_t mo, int np_, ord_t po_);
const desc_t* mad_desc_newvpo(int nv, ord_t mo, int np_, ord_t po_, const ord_t no_[]);
void          mad_desc_del(const desc_t *d_);
int           mad_desc_getnv(const desc_t *d, ord_t *mo_, int *np_, ord_t *po_);
ord_t         mad_desc_maxord(const desc_t *d, int nn, ord_t no_[]);
ssz_t         mad_desc_maxlen(const desc_t *d, ord_t mo);
log_t         mad_desc_isvalidm(const desc_t *d, ssz_t n, const ord_t m[]);
idx_t         mad_desc_idxm(const desc_t *d, ssz_t n, const ord_t m[]);
ord_t         mad_desc_mono(const desc_t *d, idx_t i, ssz_t n, ord_t m_[], ord_t *p_);

tpsa_t*       mad_tpsa_newd(const desc_t *d, ord_t mo);
tpsa_t*       mad_tpsa_new(const tpsa_t *t, ord_t mo);
void          mad_tpsa_del(const tpsa_t *t);
const desc_t* mad_tpsa_desc(const tpsa_t *t);
ord_t         mad_tpsa_mo(tpsa_t *t, ord_t mo);
ssz_t         mad_tpsa_len(const tpsa_t *t, log_t hi_);
ord_t         mad_tpsa_ord(const tpsa_t *t, log_t hi_);
log_t         mad_tpsa_isnul(const tpsa_t *t);
log_t         mad_tpsa_isval(const tpsa_t *t);
log_t         mad_tpsa_isvalid(const tpsa_t *t);

void          mad_tpsa_copy(const tpsa_t *t, tpsa_t *r);
void          mad_tpsa_getord(const tpsa_t *t, tpsa_t *r, ord_t ord);
void          mad_tpsa_cutord(const tpsa_t *t, tpsa_t *r, int ord);
void          mad_tpsa_clrord(tpsa_t *t, ord_t ord);
void          mad_tpsa_setvar(tpsa_t *t, num_t v, idx_t iv, num_t scl_);
void          mad_tpsa_setprm(tpsa_t *t, num_t v, idx_t ip);
void          mad_tpsa_setval(tpsa_t *t, num_t v);
void          mad_tpsa_update(tpsa_t *t);
void          mad_tpsa_clear(tpsa_t *t);

ord_t         mad_tpsa_mono(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_);
idx_t         mad_tpsa_idxm(const tpsa_t *t, ssz_t n, const ord_t m[]);
idx_t         mad_tpsa_cycle(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], num_t *v_);
num_t         mad_tpsa_geti(const tpsa_t *t, idx_t i);
num_t         mad_tpsa_getm(const tpsa_t *t, ssz_t n, const ord_t m[]);
void          mad_tpsa_seti(tpsa_t *t, idx_t i, num_t a, num_t b);
void          mad_tpsa_setm(tpsa_t *t, ssz_t n, const ord_t m[], num_t a, num_t b);
void          mad_tpsa_getv(const tpsa_t *t, idx_t i, ssz_t n, num_t v[]);
void          mad_tpsa_setv(tpsa_t *t, idx_t i, ssz_t n, const num_t v[]);

log_t         mad_tpsa_equ(const tpsa_t *a, const tpsa_t *b, num_t tol_);
void          mad_tpsa_dif(const tpsa_t *a, const tpsa_t *b, tpsa_t *c);
void          mad_tpsa_add(const tpsa_t *a, const tpsa_t *b, tpsa_t *c);
void          mad_tpsa_sub(const tpsa_t *a, const tpsa_t *b, tpsa_t *c);
void          mad_tpsa_mul(const tpsa_t *a, const tpsa_t *b, tpsa_t *c);
void          mad_tpsa_div(const tpsa_t *a, const tpsa_t *b, tpsa_t *c);
void          mad_tpsa_pow(const tpsa_t *a, const tpsa_t *b, tpsa_t *c);
void          mad_tpsa_powi(const tpsa_t *a, int n, tpsa_t *c);
void          mad_tpsa_pown(const tpsa_t *a, num_t v, tpsa_t *c);

num_t         mad_tpsa_nrm(const tpsa_t *a);
void          mad_tpsa_unit(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_abs(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_sqrt(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_exp(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_log(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_sincos(const tpsa_t *a, tpsa_t *s, tpsa_t *c);
void          mad_tpsa_sin(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_cos(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_tan(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_cot(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_sinc(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_sincosh(const tpsa_t *a, tpsa_t *s, tpsa_t *c);
void          mad_tpsa_sinh(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_cosh(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_tanh(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_coth(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_sinhc(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_asin(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_acos(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_atan(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_acot(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_asinc(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_asinh(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_acosh(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_atanh(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_acoth(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_asinhc(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_erf(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_erfc(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_erfcx(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_erfi(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_wf(const tpsa_t *a, tpsa_t *c);
void          mad_tpsa_scl(const tpsa_t *a, num_t v, tpsa_t *c);
void          mad_tpsa_divn(const tpsa_t *a, num_t v, tpsa_t *c);
void          mad_tpsa_inv(const tpsa_t *a, num_t v, tpsa_t *c);
void          mad_tpsa_invsqrt(const tpsa_t *a, num_t v, tpsa_t *c);
void          mad_tpsa_atan2(const tpsa_t *y, const tpsa_t *x, tpsa_t *r);
void          mad_tpsa_hypot(const tpsa_t *x, const tpsa_t *y, tpsa_t *r);
void          mad_tpsa_hypot3(const tpsa_t *x, const tpsa_t *y, const tpsa_t *z, tpsa_t *r);

void          mad_tpsa_integ(const tpsa_t *a, tpsa_t *c, idx_t iv);
void          mad_tpsa_deriv(const tpsa_t *a, tpsa_t *c, idx_t iv);
void          mad_tpsa_derivm(const tpsa_t *a, tpsa_t *c, ssz_t n, const ord_t m[]);
void          mad_tpsa_poisbra(const tpsa_t *a, const tpsa_t *b, tpsa_t *c, int nv);

ord_t         mad_tpsa_mord(ssz_t na, const tpsa_t *ma[], log_t hi);
num_t         mad_tpsa_mnrm(ssz_t na, const tpsa_t *ma[]);
void          mad_tpsa_minv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[]);
void          mad_tpsa_compose(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[]);
void          mad_tpsa_translate(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], tpsa_t *mc[]);
void          mad_tpsa_eval(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], num_t tc[]);

#ifdef __cplusplus
}
#endif

#endif
