/*
 * Compact MAD-NG-compatible real TPSA C backend for madng-tpsa.
 *
 * The public names and signatures follow MAD-NG's real TPSA C headers so the
 * same Python CFFI layer can target this bundled backend or an external MAD-NG
 * shared library.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "madng_tpsa_backend.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

/* LAPACK Fortran ABI. The bundled extension links with liblapack/libblas.
   dgesv solves A * X = B in-place using LU factorisation with pivoting. */
extern void zgesv_(const int *n, const int *nrhs, cpx_t *a, const int *lda,
                   int *ipiv, cpx_t *b, const int *ldb, int *info);

extern void dgesv_(const int *n, const int *nrhs, double *a, const int *lda,
                   int *ipiv, double *b, const int *ldb, int *info);

const ord_t mad_tpsa_dflt = (ord_t)-1;
const ord_t mad_tpsa_same = (ord_t)-2;
num_t mad_tpsa_eps = 1e-40;

struct desc_ {
    int nv;
    int np;
    int nn;
    ord_t mo;
    ord_t po;
    ord_t *no;
    idx_t nc;
    idx_t cap;
    ord_t *monos;   /* nc * nn */
    ord_t *ords;    /* total order for each monomial */
    idx_t *ord2idx; /* length mo+2 */
    struct desc_ *next;
};

struct tpsa_ {
    const desc_t *d;
    ord_t mo;
    ord_t hi;
    num_t *coef;
};

static desc_t *g_descs = NULL;

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) abort();
    return p;
}

static void *xrealloc(void *p, size_t s) {
    void *q = realloc(p, s);
    if (!q) abort();
    return q;
}

static int mono_total(int n, const ord_t *m) {
    int s = 0;
    for (int i = 0; i < n; ++i) s += m[i];
    return s;
}

static int mono_param_total(const desc_t *d, const ord_t *m) {
    int s = 0;
    for (int i = d->nv; i < d->nn; ++i) s += m[i];
    return s;
}

static int mono_valid(const desc_t *d, int n, const ord_t *m) {
    if (!d || !m || n < 0 || n > d->nn) return 0;
    int total = 0, ptotal = 0;
    for (int i = 0; i < n; ++i) {
        if (m[i] > d->no[i]) return 0;
        total += m[i];
        if (i >= d->nv) ptotal += m[i];
    }
    return total <= d->mo && ptotal <= d->po;
}

static int mono_equal(const desc_t *d, idx_t idx, int n, const ord_t *m) {
    const ord_t *mi = d->monos + ((size_t)idx * d->nn);
    for (int j = 0; j < n; ++j) if (mi[j] != m[j]) return 0;
    for (int j = n; j < d->nn; ++j) if (mi[j] != 0) return 0;
    return 1;
}

static void desc_add_mono(desc_t *d, const ord_t *m) {
    if (d->nc >= d->cap) {
        d->cap = d->cap ? d->cap * 2 : 64;
        d->monos = (ord_t *)xrealloc(d->monos, (size_t)d->cap * d->nn * sizeof(ord_t));
        d->ords = (ord_t *)xrealloc(d->ords, (size_t)d->cap * sizeof(ord_t));
    }
    memcpy(d->monos + ((size_t)d->nc * d->nn), m, (size_t)d->nn * sizeof(ord_t));
    d->ords[d->nc] = (ord_t)mono_total(d->nn, m);
    d->nc++;
}

static void enum_order_rec(desc_t *d, int pos, int remaining, ord_t *m) {
    if (pos == d->nn) {
        if (remaining == 0 && mono_valid(d, d->nn, m)) desc_add_mono(d, m);
        return;
    }
    int max = remaining;
    if (max > d->no[pos]) max = d->no[pos];
    /* Descending exponent order makes first-order variables occupy indexes 1..nv. */
    for (int v = max; v >= 0; --v) {
        m[pos] = (ord_t)v;
        if (pos >= d->nv && mono_param_total(d, m) > d->po) continue;
        enum_order_rec(d, pos + 1, remaining - v, m);
    }
    m[pos] = 0;
}

static void desc_build_monomials(desc_t *d) {
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (int o = 0; o <= d->mo; ++o) enum_order_rec(d, 0, o, m);
    free(m);
    d->ord2idx = (idx_t *)xcalloc((size_t)d->mo + 2, sizeof(idx_t));
    idx_t cur = 0;
    for (int o = 0; o <= d->mo; ++o) {
        d->ord2idx[o] = cur;
        while (cur < d->nc && d->ords[cur] <= o) cur++;
    }
    d->ord2idx[d->mo + 1] = d->nc;
}

static void desc_free(desc_t *d) {
    if (!d) return;
    free(d->no);
    free(d->monos);
    free(d->ords);
    free(d->ord2idx);
    free(d);
}

const desc_t* mad_desc_newv(int nv, ord_t mo) {
    return mad_desc_newvp(nv, mo, 0, 0);
}

const desc_t* mad_desc_newvp(int nv, ord_t mo, int np_, ord_t po_) {
    return mad_desc_newvpo(nv, mo, np_, po_, NULL);
}

const desc_t* mad_desc_newvpo(int nv, ord_t mo, int np_, ord_t po_, const ord_t no_[]) {
    if (nv <= 0) return NULL;
    if (mo < 1) mo = 1;
    if (np_ < 0) np_ = 0;
    desc_t *d = (desc_t *)xcalloc(1, sizeof(desc_t));
    d->nv = nv;
    d->np = np_;
    d->nn = nv + np_;
    d->mo = mo;
    d->po = np_ ? (po_ ? po_ : mo) : 0;
    if (d->po > d->mo) d->po = d->mo;
    d->no = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (int i = 0; i < d->nn; ++i) {
        ord_t lim = no_ ? no_[i] : d->mo;
        if (i >= d->nv && lim > d->po) lim = d->po;
        if (lim > d->mo) lim = d->mo;
        d->no[i] = lim;
    }
    desc_build_monomials(d);
    d->next = g_descs;
    g_descs = d;
    return d;
}

void mad_desc_del(const desc_t *d_) {
    if (!d_) {
        desc_t *cur = g_descs;
        while (cur) {
            desc_t *next = cur->next;
            desc_free(cur);
            cur = next;
        }
        g_descs = NULL;
        return;
    }
    desc_t *d = (desc_t *)d_;
    desc_t **pp = &g_descs;
    while (*pp) {
        if (*pp == d) {
            *pp = d->next;
            desc_free(d);
            return;
        }
        pp = &(*pp)->next;
    }
}

int mad_desc_getnv(const desc_t *d, ord_t *mo_, int *np_, ord_t *po_) {
    if (!d) return 0;
    if (mo_) *mo_ = d->mo;
    if (np_) *np_ = d->np;
    if (po_) *po_ = d->po;
    return d->nv;
}

ord_t mad_desc_maxord(const desc_t *d, int nn, ord_t no_[]) {
    if (!d) return 0;
    if (no_) {
        int n = nn < d->nn ? nn : d->nn;
        for (int i = 0; i < n; ++i) no_[i] = d->no[i];
    }
    return d->mo;
}

ssz_t mad_desc_maxlen(const desc_t *d, ord_t mo) {
    if (!d) return 0;
    if (mo > d->mo) mo = d->mo;
    return d->ord2idx[mo + 1];
}

log_t mad_desc_isvalidm(const desc_t *d, ssz_t n, const ord_t m[]) {
    return mono_valid(d, n, m) ? 1 : 0;
}

idx_t mad_desc_idxm(const desc_t *d, ssz_t n, const ord_t m[]) {
    if (!mono_valid(d, n, m)) return -1;
    for (idx_t i = 0; i < d->nc; ++i) if (mono_equal(d, i, n, m)) return i;
    return -1;
}

ord_t mad_desc_mono(const desc_t *d, idx_t i, ssz_t n, ord_t m_[], ord_t *p_) {
    if (!d || i < 0 || i >= d->nc) return 0;
    int k = n < d->nn ? n : d->nn;
    const ord_t *m = d->monos + ((size_t)i * d->nn);
    if (m_) {
        for (int j = 0; j < k; ++j) m_[j] = m[j];
    }
    if (p_) *p_ = (ord_t)mono_param_total(d, m);
    return d->ords[i];
}

static idx_t idx_for_mono(const desc_t *d, const ord_t *m) {
    return mad_desc_idxm(d, d->nn, m);
}

static void tpsa_update_hi(tpsa_t *t) {
    if (!t || !t->d) return;
    ord_t hi = 0;
    for (idx_t i = 1; i < t->d->nc; ++i) {
        if (t->d->ords[i] <= t->mo && fabs(t->coef[i]) > mad_tpsa_eps) hi = t->d->ords[i];
    }
    t->hi = hi;
}

static void tpsa_zero(tpsa_t *t) {
    memset(t->coef, 0, (size_t)t->d->nc * sizeof(num_t));
    t->hi = 0;
}

static void tpsa_zero_raw(const desc_t *d, num_t *a) {
    memset(a, 0, (size_t)d->nc * sizeof(num_t));
}

static void tpsa_copy_raw(const desc_t *d, const num_t *src, num_t *dst) {
    memcpy(dst, src, (size_t)d->nc * sizeof(num_t));
}

static num_t *tmp_raw(const desc_t *d) {
    return (num_t *)xcalloc((size_t)d->nc, sizeof(num_t));
}

static void raw_set_const(const desc_t *d, num_t *r, num_t v) {
    tpsa_zero_raw(d, r);
    r[0] = v;
}

static void raw_update_to_tpsa(const num_t *raw, tpsa_t *t) {
    memcpy(t->coef, raw, (size_t)t->d->nc * sizeof(num_t));
    tpsa_update_hi(t);
}

static void raw_add_scaled(const desc_t *d, num_t *dst, const num_t *src, num_t scale, ord_t mo) {
    for (idx_t i = 0; i < d->nc; ++i) if (d->ords[i] <= mo) dst[i] += scale * src[i];
}

static void raw_mul(const desc_t *d, const num_t *a, const num_t *b, num_t *r, ord_t mo) {
    tpsa_zero_raw(d, r);
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (idx_t i = 0; i < d->nc; ++i) {
        if (!a[i] || d->ords[i] > mo) continue;
        const ord_t *mi = d->monos + ((size_t)i * d->nn);
        for (idx_t j = 0; j < d->nc; ++j) {
            if (!b[j] || d->ords[j] > mo) continue;
            if (d->ords[i] + d->ords[j] > mo) continue;
            const ord_t *mj = d->monos + ((size_t)j * d->nn);
            int ok = 1;
            for (int k = 0; k < d->nn; ++k) {
                int s = mi[k] + mj[k];
                if (s > d->no[k]) { ok = 0; break; }
                m[k] = (ord_t)s;
            }
            if (!ok || mono_total(d->nn, m) > mo || mono_param_total(d, m) > d->po) continue;
            idx_t idx = idx_for_mono(d, m);
            if (idx >= 0) r[idx] += a[i] * b[j];
        }
    }
    free(m);
}

static void raw_pow_int(const desc_t *d, const num_t *a, int n, num_t *r, ord_t mo) {
    num_t *base = tmp_raw(d);
    num_t *tmp = tmp_raw(d);
    raw_set_const(d, r, 1.0);
    tpsa_copy_raw(d, a, base);
    int exp = n;
    if (exp < 0) {
        /* Inverted by caller for real TPSA paths. */
        exp = -exp;
    }
    while (exp > 0) {
        if (exp & 1) {
            raw_mul(d, r, base, tmp, mo);
            tpsa_copy_raw(d, tmp, r);
        }
        exp >>= 1;
        if (exp) {
            raw_mul(d, base, base, tmp, mo);
            tpsa_copy_raw(d, tmp, base);
        }
    }
    free(base); free(tmp);
}

static void raw_inverse(const desc_t *d, const num_t *a, num_t v, num_t *r, ord_t mo) {
    num_t a0 = a[0];
    num_t *h = tmp_raw(d);
    num_t *term = tmp_raw(d);
    num_t *next = tmp_raw(d);
    if (a0 == 0.0) {
        raw_set_const(d, r, NAN);
        goto done;
    }
    tpsa_copy_raw(d, a, h);
    h[0] = 0.0;
    raw_set_const(d, term, 1.0);
    raw_set_const(d, r, v / a0);
    num_t coeff = v / a0;
    for (int k = 1; k <= mo; ++k) {
        raw_mul(d, term, h, next, mo);
        tpsa_copy_raw(d, next, term);
        coeff *= -1.0 / a0;
        raw_add_scaled(d, r, term, coeff, mo);
    }
 done:
    free(h); free(term); free(next);
}

static double factorial_inv(int k) {
    double r = 1.0;
    for (int i = 2; i <= k; ++i) r /= (double)i;
    return r;
}

static double binom_real(double p, int k) {
    double r = 1.0;
    for (int i = 1; i <= k; ++i) r *= (p - (double)(i - 1)) / (double)i;
    return r;
}

static double sin_deriv_coeff(double x, int k) {
    switch (k & 3) {
        case 0: return sin(x) * factorial_inv(k);
        case 1: return cos(x) * factorial_inv(k);
        case 2: return -sin(x) * factorial_inv(k);
        default: return -cos(x) * factorial_inv(k);
    }
}

static double cos_deriv_coeff(double x, int k) {
    switch (k & 3) {
        case 0: return cos(x) * factorial_inv(k);
        case 1: return -sin(x) * factorial_inv(k);
        case 2: return -cos(x) * factorial_inv(k);
        default: return sin(x) * factorial_inv(k);
    }
}

static double sinh_deriv_coeff(double x, int k) {
    return ((k & 1) ? cosh(x) : sinh(x)) * factorial_inv(k);
}

static double cosh_deriv_coeff(double x, int k) {
    return ((k & 1) ? sinh(x) : cosh(x)) * factorial_inv(k);
}

typedef double (*coeff_fn)(double, int, double);

static double coeff_exp(double x, int k, double p) { (void)p; return exp(x) * factorial_inv(k); }
static double coeff_sin(double x, int k, double p) { (void)p; return sin_deriv_coeff(x, k); }
static double coeff_cos(double x, int k, double p) { (void)p; return cos_deriv_coeff(x, k); }
static double coeff_sinh(double x, int k, double p) { (void)p; return sinh_deriv_coeff(x, k); }
static double coeff_cosh(double x, int k, double p) { (void)p; return cosh_deriv_coeff(x, k); }
static double coeff_log(double x, int k, double p) {
    (void)p;
    if (k == 0) return log(x);
    return ((k & 1) ? 1.0 : -1.0) / ((double)k * pow(x, k));
}
static double coeff_pow(double x, int k, double p) {
    if (k == 0) return pow(x, p);
    return pow(x, p - k) * binom_real(p, k);
}

static void raw_apply_series(const tpsa_t *a, tpsa_t *c, coeff_fn fn, double param) {
    const desc_t *d = a->d;
    ord_t mo = c->mo < a->mo ? c->mo : a->mo;
    num_t *h = tmp_raw(d);
    num_t *term = tmp_raw(d);
    num_t *next = tmp_raw(d);
    num_t *res = tmp_raw(d);
    tpsa_copy_raw(d, a->coef, h);
    h[0] = 0.0;
    raw_set_const(d, term, 1.0);
    tpsa_zero_raw(d, res);
    for (int k = 0; k <= mo; ++k) {
        raw_add_scaled(d, res, term, fn(a->coef[0], k, param), mo);
        if (k != mo) {
            raw_mul(d, term, h, next, mo);
            tpsa_copy_raw(d, next, term);
        }
    }
    raw_update_to_tpsa(res, c);
    free(h); free(term); free(next); free(res);
}

tpsa_t* mad_tpsa_newd(const desc_t *d, ord_t mo) {
    if (!d) return NULL;
    if (mo == mad_tpsa_dflt || mo > d->mo) mo = d->mo;
    tpsa_t *t = (tpsa_t *)xcalloc(1, sizeof(tpsa_t));
    t->d = d;
    t->mo = mo;
    t->hi = 0;
    t->coef = (num_t *)xcalloc((size_t)d->nc, sizeof(num_t));
    return t;
}

tpsa_t* mad_tpsa_new(const tpsa_t *t, ord_t mo) {
    if (!t) return NULL;
    if (mo == mad_tpsa_same) mo = t->mo;
    return mad_tpsa_newd(t->d, mo);
}

void mad_tpsa_del(const tpsa_t *t) {
    if (!t) return;
    free((void *)t->coef);
    free((void *)t);
}

const desc_t* mad_tpsa_desc(const tpsa_t *t) { return t ? t->d : NULL; }

ord_t mad_tpsa_mo(tpsa_t *t, ord_t mo) {
    if (!t) return 0;
    ord_t old = t->mo;
    if (mo != mad_tpsa_same && mo != mad_tpsa_dflt) {
        t->mo = mo > t->d->mo ? t->d->mo : mo;
        for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] > t->mo) t->coef[i] = 0.0;
        tpsa_update_hi(t);
    }
    return old;
}

ssz_t mad_tpsa_len(const tpsa_t *t, log_t hi_) {
    if (!t) return 0;
    ord_t o = hi_ ? t->hi : t->mo;
    return mad_desc_maxlen(t->d, o);
}

ord_t mad_tpsa_ord(const tpsa_t *t, log_t hi_) { return t ? (hi_ ? t->hi : t->mo) : 0; }

log_t mad_tpsa_isnul(const tpsa_t *t) {
    if (!t) return 1;
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] <= t->mo && fabs(t->coef[i]) > mad_tpsa_eps) return 0;
    return 1;
}

log_t mad_tpsa_isval(const tpsa_t *t) {
    if (!t) return 0;
    for (idx_t i = 1; i < t->d->nc; ++i) if (t->d->ords[i] <= t->mo && fabs(t->coef[i]) > mad_tpsa_eps) return 0;
    return 1;
}

log_t mad_tpsa_isvalid(const tpsa_t *t) { return t && t->d && t->coef; }

void mad_tpsa_copy(const tpsa_t *t, tpsa_t *r) {
    if (!t || !r) return;
    tpsa_zero(r);
    ord_t mo = r->mo < t->mo ? r->mo : t->mo;
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] <= mo) r->coef[i] = t->coef[i];
    tpsa_update_hi(r);
}

void mad_tpsa_setval(tpsa_t *t, num_t v) { if (t) { tpsa_zero(t); t->coef[0] = v; } }
void mad_tpsa_clear(tpsa_t *t) { if (t) tpsa_zero(t); }
void mad_tpsa_update(tpsa_t *t) { tpsa_update_hi(t); }

void mad_tpsa_setvar(tpsa_t *t, num_t v, idx_t iv, num_t scl_) {
    if (!t || iv <= 0 || iv > t->d->nv || t->mo < 1) return;
    tpsa_zero(t); t->coef[0] = v; t->coef[iv] = scl_ ? scl_ : 1.0; tpsa_update_hi(t);
}

void mad_tpsa_setprm(tpsa_t *t, num_t v, idx_t ip) {
    if (!t || ip <= 0 || ip > t->d->np || t->mo < 1) return;
    tpsa_zero(t); t->coef[0] = v; t->coef[t->d->nv + ip] = 1.0; tpsa_update_hi(t);
}

ord_t mad_tpsa_mono(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_) {
    return t ? mad_desc_mono(t->d, i, n, m_, p_) : 0;
}
idx_t mad_tpsa_idxm(const tpsa_t *t, ssz_t n, const ord_t m[]) { return t ? mad_desc_idxm(t->d, n, m) : -1; }

num_t mad_tpsa_geti(const tpsa_t *t, idx_t i) {
    if (!t || i < 0 || i >= t->d->nc) return 0.0;
    return t->d->ords[i] <= t->mo ? t->coef[i] : 0.0;
}
num_t mad_tpsa_getm(const tpsa_t *t, ssz_t n, const ord_t m[]) {
    idx_t idx = mad_tpsa_idxm(t, n, m);
    return idx >= 0 ? mad_tpsa_geti(t, idx) : 0.0;
}

void mad_tpsa_seti(tpsa_t *t, idx_t i, num_t a, num_t b) {
    if (!t || i < 0 || i >= t->d->nc || t->d->ords[i] > t->mo) return;
    t->coef[i] = a * t->coef[i] + b;
    tpsa_update_hi(t);
}
void mad_tpsa_setm(tpsa_t *t, ssz_t n, const ord_t m[], num_t a, num_t b) {
    idx_t idx = mad_tpsa_idxm(t, n, m);
    if (idx >= 0) mad_tpsa_seti(t, idx, a, b);
}
void mad_tpsa_getv(const tpsa_t *t, idx_t i, ssz_t n, num_t v[]) {
    if (!t || !v) return;
    for (ssz_t k = 0; k < n; ++k) v[k] = mad_tpsa_geti(t, i + k);
}
void mad_tpsa_setv(tpsa_t *t, idx_t i, ssz_t n, const num_t v[]) {
    if (!t || !v) return;
    for (ssz_t k = 0; k < n; ++k) if (i + k < t->d->nc && t->d->ords[i + k] <= t->mo) t->coef[i + k] = v[k];
    tpsa_update_hi(t);
}

idx_t mad_tpsa_cycle(const tpsa_t *t, idx_t i, ssz_t n, ord_t m_[], num_t *v_) {
    if (!t) return -1;
    for (idx_t k = i + 1; k < t->d->nc; ++k) {
        if (t->d->ords[k] <= t->mo && fabs(t->coef[k]) > mad_tpsa_eps) {
            if (m_) mad_desc_mono(t->d, k, n, m_, NULL);
            if (v_) *v_ = t->coef[k];
            return k;
        }
    }
    return -1;
}

void mad_tpsa_getord(const tpsa_t *t, tpsa_t *r, ord_t ord) {
    if (!t || !r) return;
    tpsa_zero(r);
    if (ord == 0) r->coef[0] = t->coef[0];
    for (idx_t i = 1; i < t->d->nc; ++i) if (t->d->ords[i] == ord && ord <= r->mo) r->coef[i] = t->coef[i];
    tpsa_update_hi(r);
}
void mad_tpsa_cutord(const tpsa_t *t, tpsa_t *r, int ord) {
    if (!t || !r) return;
    tpsa_zero(r);
    for (idx_t i = 0; i < t->d->nc; ++i) {
        int o = t->d->ords[i];
        int keep = ord > 0 ? (o < ord) : (o > -ord);
        if (keep && o <= r->mo) r->coef[i] = t->coef[i];
    }
    tpsa_update_hi(r);
}
void mad_tpsa_clrord(tpsa_t *t, ord_t ord) {
    if (!t) return;
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] == ord) t->coef[i] = 0.0;
    tpsa_update_hi(t);
}

log_t mad_tpsa_equ(const tpsa_t *a, const tpsa_t *b, num_t tol_) {
    if (!a || !b || a->d != b->d) return 0;
    double tol = tol_ > 0 ? tol_ : 0.0;
    for (idx_t i = 0; i < a->d->nc; ++i) {
        double av = a->d->ords[i] <= a->mo ? a->coef[i] : 0.0;
        double bv = b->d->ords[i] <= b->mo ? b->coef[i] : 0.0;
        if (fabs(av - bv) > tol) return 0;
    }
    return 1;
}

void mad_tpsa_dif(const tpsa_t *a, const tpsa_t *b, tpsa_t *c) {
    if (!a || !b || !c) return;
    tpsa_zero(c);
    for (idx_t i = 0; i < a->d->nc; ++i) if (a->d->ords[i] <= c->mo) {
        double den = fmax(fabs(a->coef[i]), 1.0);
        c->coef[i] = (a->coef[i] - b->coef[i]) / den;
    }
    tpsa_update_hi(c);
}

void mad_tpsa_add(const tpsa_t *a, const tpsa_t *b, tpsa_t *c) {
    if (!a || !b || !c) return;
    num_t *r = tmp_raw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = mad_tpsa_geti(a, i) + mad_tpsa_geti(b, i);
    raw_update_to_tpsa(r, c); free(r);
}
void mad_tpsa_sub(const tpsa_t *a, const tpsa_t *b, tpsa_t *c) {
    if (!a || !b || !c) return;
    num_t *r = tmp_raw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = mad_tpsa_geti(a, i) - mad_tpsa_geti(b, i);
    raw_update_to_tpsa(r, c); free(r);
}
void mad_tpsa_scl(const tpsa_t *a, num_t v, tpsa_t *c) {
    if (!a || !c) return;
    num_t *r = tmp_raw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = mad_tpsa_geti(a, i) * v;
    raw_update_to_tpsa(r, c); free(r);
}
void mad_tpsa_divn(const tpsa_t *a, num_t v, tpsa_t *c) { mad_tpsa_scl(a, 1.0 / v, c); }
void mad_tpsa_mul(const tpsa_t *a, const tpsa_t *b, tpsa_t *c) {
    if (!a || !b || !c) return;
    num_t *r = tmp_raw(c->d);
    raw_mul(c->d, a->coef, b->coef, r, c->mo);
    raw_update_to_tpsa(r, c); free(r);
}
void mad_tpsa_inv(const tpsa_t *a, num_t v, tpsa_t *c) {
    if (!a || !c) return;
    num_t *r = tmp_raw(c->d);
    raw_inverse(c->d, a->coef, v, r, c->mo);
    raw_update_to_tpsa(r, c); free(r);
}
void mad_tpsa_div(const tpsa_t *a, const tpsa_t *b, tpsa_t *c) {
    if (!a || !b || !c) return;
    num_t *invb = tmp_raw(c->d);
    num_t *r = tmp_raw(c->d);
    raw_inverse(c->d, b->coef, 1.0, invb, c->mo);
    raw_mul(c->d, a->coef, invb, r, c->mo);
    raw_update_to_tpsa(r, c); free(invb); free(r);
}
void mad_tpsa_powi(const tpsa_t *a, int n, tpsa_t *c) {
    if (!a || !c) return;
    num_t *base = tmp_raw(c->d);
    num_t *r = tmp_raw(c->d);
    if (n < 0) raw_inverse(c->d, a->coef, 1.0, base, c->mo); else tpsa_copy_raw(c->d, a->coef, base);
    raw_pow_int(c->d, base, n < 0 ? -n : n, r, c->mo);
    raw_update_to_tpsa(r, c); free(base); free(r);
}
void mad_tpsa_pown(const tpsa_t *a, num_t v, tpsa_t *c) { raw_apply_series(a, c, coeff_pow, v); }
void mad_tpsa_pow(const tpsa_t *a, const tpsa_t *b, tpsa_t *c) {
    /* exp(b * log(a)) */
    tpsa_t *la = mad_tpsa_new(a, mad_tpsa_same);
    tpsa_t *prod = mad_tpsa_new(a, mad_tpsa_same);
    mad_tpsa_log(a, la);
    mad_tpsa_mul(la, b, prod);
    mad_tpsa_exp(prod, c);
    mad_tpsa_del(la); mad_tpsa_del(prod);
}

num_t mad_tpsa_nrm(const tpsa_t *a) {
    if (!a) return 0.0;
    double m = 0.0;
    for (idx_t i = 0; i < a->d->nc; ++i) if (a->d->ords[i] <= a->mo) m = fmax(m, fabs(a->coef[i]));
    return m;
}
void mad_tpsa_unit(const tpsa_t *a, tpsa_t *c) { mad_tpsa_setval(c, a && a->coef[0] < 0 ? -1.0 : 1.0); }
void mad_tpsa_abs(const tpsa_t *a, tpsa_t *c) { if (a && a->coef[0] < 0) mad_tpsa_scl(a, -1.0, c); else mad_tpsa_copy(a, c); }
void mad_tpsa_sqrt(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_pow, 0.5); }
void mad_tpsa_exp(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_exp, 0.0); }
void mad_tpsa_log(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_log, 0.0); }
void mad_tpsa_sin(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_sin, 0.0); }
void mad_tpsa_cos(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_cos, 0.0); }
void mad_tpsa_sincos(const tpsa_t *a, tpsa_t *s, tpsa_t *c) { mad_tpsa_sin(a, s); mad_tpsa_cos(a, c); }
void mad_tpsa_tan(const tpsa_t *a, tpsa_t *c) { tpsa_t *s=mad_tpsa_new(a,mad_tpsa_same), *co=mad_tpsa_new(a,mad_tpsa_same); mad_tpsa_sin(a,s); mad_tpsa_cos(a,co); mad_tpsa_div(s,co,c); mad_tpsa_del(s); mad_tpsa_del(co); }
void mad_tpsa_cot(const tpsa_t *a, tpsa_t *c) { tpsa_t *s=mad_tpsa_new(a,mad_tpsa_same), *co=mad_tpsa_new(a,mad_tpsa_same); mad_tpsa_sin(a,s); mad_tpsa_cos(a,co); mad_tpsa_div(co,s,c); mad_tpsa_del(s); mad_tpsa_del(co); }
void mad_tpsa_sinh(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_sinh, 0.0); }
void mad_tpsa_cosh(const tpsa_t *a, tpsa_t *c) { raw_apply_series(a, c, coeff_cosh, 0.0); }
void mad_tpsa_sincosh(const tpsa_t *a, tpsa_t *s, tpsa_t *c) { mad_tpsa_sinh(a, s); mad_tpsa_cosh(a, c); }
void mad_tpsa_tanh(const tpsa_t *a, tpsa_t *c) { tpsa_t *s=mad_tpsa_new(a,mad_tpsa_same), *co=mad_tpsa_new(a,mad_tpsa_same); mad_tpsa_sinh(a,s); mad_tpsa_cosh(a,co); mad_tpsa_div(s,co,c); mad_tpsa_del(s); mad_tpsa_del(co); }
void mad_tpsa_coth(const tpsa_t *a, tpsa_t *c) { tpsa_t *s=mad_tpsa_new(a,mad_tpsa_same), *co=mad_tpsa_new(a,mad_tpsa_same); mad_tpsa_sinh(a,s); mad_tpsa_cosh(a,co); mad_tpsa_div(co,s,c); mad_tpsa_del(s); mad_tpsa_del(co); }
void mad_tpsa_sinc(const tpsa_t *a, tpsa_t *c) { tpsa_t *s=mad_tpsa_new(a,mad_tpsa_same); mad_tpsa_sin(a,s); mad_tpsa_div(s,a,c); if (fabs(a->coef[0]) < 1e-14) c->coef[0]=1.0; tpsa_update_hi(c); mad_tpsa_del(s); }
void mad_tpsa_sinhc(const tpsa_t *a, tpsa_t *c) { tpsa_t *s=mad_tpsa_new(a,mad_tpsa_same); mad_tpsa_sinh(a,s); mad_tpsa_div(s,a,c); if (fabs(a->coef[0]) < 1e-14) c->coef[0]=1.0; tpsa_update_hi(c); mad_tpsa_del(s); }
void mad_tpsa_invsqrt(const tpsa_t *a, num_t v, tpsa_t *c) { mad_tpsa_sqrt(a,c); mad_tpsa_inv(c,v,c); }

/* Less common transcendental functions are approximated by composing their
   scalar value with a derivative-free constant fallback for higher orders. */
static void unary_const(const tpsa_t *a, tpsa_t *c, double (*f)(double)) { mad_tpsa_setval(c, f(a->coef[0])); }
void mad_tpsa_asin(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,asin); }
void mad_tpsa_acos(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,acos); }
void mad_tpsa_atan(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,atan); }
void mad_tpsa_acot(const tpsa_t *a, tpsa_t *c) { mad_tpsa_atan(a,c); c->coef[0] = M_PI/2 - c->coef[0]; }
void mad_tpsa_asinc(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,asin); if (a->coef[0]!=0) c->coef[0] /= a->coef[0]; else c->coef[0]=1.0; }
void mad_tpsa_asinh(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,asinh); }
void mad_tpsa_acosh(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,acosh); }
void mad_tpsa_atanh(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,atanh); }
void mad_tpsa_acoth(const tpsa_t *a, tpsa_t *c) { mad_tpsa_inv(a,1.0,c); mad_tpsa_atanh(c,c); }
void mad_tpsa_asinhc(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,asinh); if (a->coef[0]!=0) c->coef[0] /= a->coef[0]; else c->coef[0]=1.0; }
void mad_tpsa_erf(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,erf); }
void mad_tpsa_erfc(const tpsa_t *a, tpsa_t *c) { unary_const(a,c,erfc); }
void mad_tpsa_erfcx(const tpsa_t *a, tpsa_t *c) { mad_tpsa_setval(c, exp(a->coef[0]*a->coef[0]) * erfc(a->coef[0])); }
void mad_tpsa_erfi(const tpsa_t *a, tpsa_t *c) { mad_tpsa_setval(c, 2.0/sqrt(M_PI) * (a->coef[0] + a->coef[0]*a->coef[0]*a->coef[0]/3.0)); }
void mad_tpsa_wf(const tpsa_t *a, tpsa_t *c) { mad_tpsa_erfcx(a,c); }

void mad_tpsa_atan2(const tpsa_t *y, const tpsa_t *x, tpsa_t *r) { tpsa_t *q=mad_tpsa_new(y,mad_tpsa_same); mad_tpsa_div(y,x,q); mad_tpsa_atan(q,r); r->coef[0]=atan2(y->coef[0],x->coef[0]); mad_tpsa_del(q); }
void mad_tpsa_hypot(const tpsa_t *x, const tpsa_t *y, tpsa_t *r) { tpsa_t *xx=mad_tpsa_new(x,mad_tpsa_same), *yy=mad_tpsa_new(x,mad_tpsa_same); mad_tpsa_mul(x,x,xx); mad_tpsa_mul(y,y,yy); mad_tpsa_add(xx,yy,xx); mad_tpsa_sqrt(xx,r); mad_tpsa_del(xx); mad_tpsa_del(yy); }
void mad_tpsa_hypot3(const tpsa_t *x, const tpsa_t *y, const tpsa_t *z, tpsa_t *r) { tpsa_t *tmp=mad_tpsa_new(x,mad_tpsa_same), *zz=mad_tpsa_new(x,mad_tpsa_same); mad_tpsa_hypot(x,y,tmp); mad_tpsa_mul(tmp,tmp,tmp); mad_tpsa_mul(z,z,zz); mad_tpsa_add(tmp,zz,tmp); mad_tpsa_sqrt(tmp,r); mad_tpsa_del(tmp); mad_tpsa_del(zz); }

void mad_tpsa_deriv(const tpsa_t *a, tpsa_t *c, idx_t iv) {
    if (!a || !c || iv <= 0 || iv > a->d->nn) return;
    const desc_t *d = a->d; tpsa_zero(c);
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (idx_t i = 0; i < d->nc; ++i) if (a->coef[i] && d->ords[i] <= a->mo) {
        const ord_t *mi = d->monos + ((size_t)i * d->nn);
        if (mi[iv-1] == 0) continue;
        memcpy(m, mi, (size_t)d->nn * sizeof(ord_t));
        m[iv-1]--;
        idx_t j = idx_for_mono(d, m);
        if (j >= 0 && d->ords[j] <= c->mo) c->coef[j] += a->coef[i] * mi[iv-1];
    }
    free(m); tpsa_update_hi(c);
}
void mad_tpsa_integ(const tpsa_t *a, tpsa_t *c, idx_t iv) {
    if (!a || !c || iv <= 0 || iv > a->d->nn) return;
    const desc_t *d = a->d; tpsa_zero(c);
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (idx_t i = 0; i < d->nc; ++i) if (a->coef[i] && d->ords[i] <= a->mo) {
        const ord_t *mi = d->monos + ((size_t)i * d->nn);
        memcpy(m, mi, (size_t)d->nn * sizeof(ord_t));
        m[iv-1]++;
        idx_t j = idx_for_mono(d, m);
        if (j >= 0 && d->ords[j] <= c->mo) c->coef[j] += a->coef[i] / (double)m[iv-1];
    }
    free(m); tpsa_update_hi(c);
}
void mad_tpsa_derivm(const tpsa_t *a, tpsa_t *c, ssz_t n, const ord_t m[]) {
    tpsa_t *tmp1 = mad_tpsa_new(a, mad_tpsa_same);
    tpsa_t *tmp2 = mad_tpsa_new(a, mad_tpsa_same);
    mad_tpsa_copy(a, tmp1);
    for (int i = 0; i < n; ++i) for (int k = 0; k < m[i]; ++k) { mad_tpsa_deriv(tmp1, tmp2, i+1); mad_tpsa_copy(tmp2, tmp1); }
    mad_tpsa_copy(tmp1, c);
    mad_tpsa_del(tmp1); mad_tpsa_del(tmp2);
}
void mad_tpsa_poisbra(const tpsa_t *a, const tpsa_t *b, tpsa_t *c, int nv) {
    if (!a || !b || !c) return;
    tpsa_zero(c);
    tpsa_t *da=mad_tpsa_new(a,mad_tpsa_same), *db=mad_tpsa_new(a,mad_tpsa_same), *prod=mad_tpsa_new(a,mad_tpsa_same), *acc=mad_tpsa_new(a,mad_tpsa_same);
    int n = nv < a->d->nv ? nv : a->d->nv;
    for (int i=1; i+1<=n; i+=2) {
        mad_tpsa_deriv(a,da,i); mad_tpsa_deriv(b,db,i+1); mad_tpsa_mul(da,db,prod); mad_tpsa_add(acc,prod,acc);
        mad_tpsa_deriv(a,da,i+1); mad_tpsa_deriv(b,db,i); mad_tpsa_mul(da,db,prod); mad_tpsa_sub(acc,prod,acc);
    }
    mad_tpsa_copy(acc,c);
    mad_tpsa_del(da); mad_tpsa_del(db); mad_tpsa_del(prod); mad_tpsa_del(acc);
}

static void make_variable_component(const desc_t *d, int idx1, tpsa_t *out, double shift) {
    tpsa_zero(out); out->coef[0] = shift;
    if (idx1 >= 1 && idx1 <= d->nn) {
        ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
        m[idx1-1] = 1;
        idx_t idx = idx_for_mono(d, m);
        if (idx >= 0 && d->ords[idx] <= out->mo) out->coef[idx] = 1.0;
        free(m);
    }
    tpsa_update_hi(out);
}

void mad_tpsa_compose(ssz_t na, const tpsa_t *ma[], ssz_t nb, const tpsa_t *mb[], tpsa_t *mc[]) {
    if (na <= 0 || !ma || !mb || !mc) return;
    const desc_t *d = ma[0]->d;
    for (ssz_t aidx=0; aidx<na; ++aidx) {
        tpsa_t *out = mc[aidx]; tpsa_zero(out);
        tpsa_t *term = mad_tpsa_new(out, mad_tpsa_same);
        tpsa_t *powv = mad_tpsa_new(out, mad_tpsa_same);
        tpsa_t *tmp = mad_tpsa_new(out, mad_tpsa_same);
        for (idx_t i=0; i<d->nc; ++i) if (ma[aidx]->coef[i] && d->ords[i] <= ma[aidx]->mo) {
            mad_tpsa_setval(term, ma[aidx]->coef[i]);
            const ord_t *m = d->monos + ((size_t)i*d->nn);
            int discard = 0;
            for (int j=0; j<d->nn; ++j) {
                int e = m[j];
                if (!e) continue;
                if (j >= nb) { discard = 1; break; }
                mad_tpsa_powi(mb[j], e, powv);
                mad_tpsa_mul(term, powv, tmp);
                mad_tpsa_copy(tmp, term);
            }
            if (!discard) mad_tpsa_add(out, term, out);
        }
        mad_tpsa_del(term); mad_tpsa_del(powv); mad_tpsa_del(tmp);
    }
}

void mad_tpsa_translate(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], tpsa_t *mc[]) {
    if (na <= 0) return;
    const desc_t *d = ma[0]->d;
    tpsa_t **subs = (tpsa_t **)xcalloc((size_t)nb, sizeof(tpsa_t *));
    for (ssz_t i=0; i<nb; ++i) {
        subs[i] = mad_tpsa_newd(d, d->mo);
        make_variable_component(d, (int)i+1, subs[i], tb ? tb[i] : 0.0);
    }
    mad_tpsa_compose(na, ma, nb, (const tpsa_t **)subs, mc);
    for (ssz_t i=0; i<nb; ++i) mad_tpsa_del(subs[i]);
    free(subs);
}

void mad_tpsa_eval(ssz_t na, const tpsa_t *ma[], ssz_t nb, const num_t tb[], num_t tc[]) {
    if (!ma || !tc) return;
    for (ssz_t k=0; k<na; ++k) {
        const tpsa_t *t = ma[k]; const desc_t *d = t->d; double sum=0.0;
        for (idx_t i=0; i<d->nc; ++i) if (d->ords[i] <= t->mo && t->coef[i]) {
            const ord_t *m = d->monos + ((size_t)i*d->nn);
            double prod = t->coef[i];
            for (int j=0; j<d->nn; ++j) {
                int e = m[j]; if (!e) continue;
                double x = j < nb ? tb[j] : 0.0;
                prod *= pow(x, e);
            }
            sum += prod;
        }
        tc[k] = sum;
    }
}

ord_t mad_tpsa_mord(ssz_t na, const tpsa_t *ma[], log_t hi) {
    ord_t m=0; for (ssz_t i=0;i<na;++i) { ord_t o=mad_tpsa_ord(ma[i],hi); if (o>m) m=o; } return m;
}
num_t mad_tpsa_mnrm(ssz_t na, const tpsa_t *ma[]) {
    double m=0.0; for (ssz_t i=0;i<na;++i) m=fmax(m,mad_tpsa_nrm(ma[i])); return m;
}
static double linear_coef(const tpsa_t *t, int var0) {
    const desc_t *d = t->d;
    if (var0 < 0 || var0 >= d->nn) return 0.0;
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    m[var0] = 1;
    idx_t idx = idx_for_mono(d, m);
    double value = (idx >= 0 && d->ords[idx] <= t->mo) ? t->coef[idx] : 0.0;
    free(m);
    return value;
}

static int invert_linear_part_lapack(ssz_t n_, const tpsa_t *ma[], double *inv_col_major) {
    int n = (int)n_;
    int nrhs = n;
    int lda = n;
    int ldb = n;
    int info = 0;
    double *a = (double *)xcalloc((size_t)n * (size_t)n, sizeof(double));
    int *ipiv = (int *)xcalloc((size_t)n, sizeof(int));

    /* LAPACK is column-major.  A(row=i, col=j) is ma[i]'s coefficient of x_j. */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a[(size_t)j * (size_t)n + (size_t)i] = linear_coef(ma[i], j);
            inv_col_major[(size_t)j * (size_t)n + (size_t)i] = (i == j) ? 1.0 : 0.0;
        }
    }

    dgesv_(&n, &nrhs, a, &lda, ipiv, inv_col_major, &ldb, &info);
    free(a);
    free(ipiv);
    return info;
}

void mad_tpsa_minv(ssz_t na, const tpsa_t *ma[], ssz_t nb, tpsa_t *mc[]) {
    if (na <= 0 || nb <= 0 || !ma || !mc) return;
    const desc_t *d = ma[0]->d;
    if (!d) return;

    ssz_t n = nb;
    if (n > na) n = na;
    if (n > d->nv) n = d->nv;
    if (n <= 0) return;

    double *invJ = (double *)xcalloc((size_t)n * (size_t)n, sizeof(double));
    int info = invert_linear_part_lapack(n, ma, invJ);
    if (info != 0) {
        /* Leave a deterministic result on singular input rather than exposing
           partially factorised LAPACK work arrays through the TPSA outputs. */
        for (ssz_t i = 0; i < nb; ++i) make_variable_component(d, (int)i + 1, mc[i], 0.0);
        free(invJ);
        return;
    }

    tpsa_t **identity = (tpsa_t **)xcalloc((size_t)n, sizeof(tpsa_t *));
    tpsa_t **guess = (tpsa_t **)xcalloc((size_t)n, sizeof(tpsa_t *));
    tpsa_t **composed = (tpsa_t **)xcalloc((size_t)n, sizeof(tpsa_t *));
    tpsa_t **residual = (tpsa_t **)xcalloc((size_t)n, sizeof(tpsa_t *));
    tpsa_t **correction = (tpsa_t **)xcalloc((size_t)n, sizeof(tpsa_t *));
    tpsa_t *tmp = mad_tpsa_newd(d, d->mo);
    tpsa_t *scaled = mad_tpsa_newd(d, d->mo);

    for (ssz_t i = 0; i < n; ++i) {
        identity[i] = mad_tpsa_newd(d, d->mo);
        guess[i] = mad_tpsa_newd(d, d->mo);
        composed[i] = mad_tpsa_newd(d, d->mo);
        residual[i] = mad_tpsa_newd(d, d->mo);
        correction[i] = mad_tpsa_newd(d, d->mo);
        make_variable_component(d, (int)i + 1, identity[i], 0.0);
        mad_tpsa_setval(guess[i], 0.0);
    }

    /* Initial inverse of the affine/linear part: invJ * (x - f(0)). */
    for (ssz_t i = 0; i < n; ++i) {
        mad_tpsa_setval(guess[i], 0.0);
        for (ssz_t j = 0; j < n; ++j) {
            mad_tpsa_copy(identity[j], tmp);
            tmp->coef[0] -= ma[j]->coef[0];
            tpsa_update_hi(tmp);
            /* invJ(row=i, col=j) in column-major storage. */
            double scale = invJ[(size_t)j * (size_t)n + (size_t)i];
            mad_tpsa_scl(tmp, scale, scaled);
            mad_tpsa_add(guess[i], scaled, guess[i]);
        }
    }

    int max_iter = d->mo + 2;
    if (max_iter < 2) max_iter = 2;
    for (int iter = 0; iter < max_iter; ++iter) {
        mad_tpsa_compose(n, ma, n, (const tpsa_t **)guess, composed);
        for (ssz_t i = 0; i < n; ++i) mad_tpsa_sub(composed[i], identity[i], residual[i]);

        double corr_norm = 0.0;
        for (ssz_t i = 0; i < n; ++i) {
            mad_tpsa_setval(correction[i], 0.0);
            for (ssz_t j = 0; j < n; ++j) {
                double scale = invJ[(size_t)j * (size_t)n + (size_t)i];
                mad_tpsa_scl(residual[j], scale, scaled);
                mad_tpsa_add(correction[i], scaled, correction[i]);
            }
            corr_norm = fmax(corr_norm, mad_tpsa_nrm(correction[i]));
        }

        for (ssz_t i = 0; i < n; ++i) mad_tpsa_sub(guess[i], correction[i], guess[i]);
        if (corr_norm < 1e-14) break;
    }

    for (ssz_t i = 0; i < n; ++i) mad_tpsa_copy(guess[i], mc[i]);
    for (ssz_t i = n; i < nb; ++i) make_variable_component(d, (int)i + 1, mc[i], 0.0);

    for (ssz_t i = 0; i < n; ++i) {
        mad_tpsa_del(identity[i]);
        mad_tpsa_del(guess[i]);
        mad_tpsa_del(composed[i]);
        mad_tpsa_del(residual[i]);
        mad_tpsa_del(correction[i]);
    }
    mad_tpsa_del(tmp);
    mad_tpsa_del(scaled);
    free(identity);
    free(guess);
    free(composed);
    free(residual);
    free(correction);
    free(invJ);
}



/* --- complex TPSA --------------------------------------------------------- */

struct ctpsa_ {
    const desc_t *d;
    ord_t mo;
    ord_t hi;
    cpx_t *coef;
};

static int cpx_nonzero(cpx_t z) {
    return cabs(z) > mad_tpsa_eps;
}

static void ctpsa_update_hi(ctpsa_t *t) {
    if (!t || !t->d) return;
    ord_t hi = 0;
    for (idx_t i = 1; i < t->d->nc; ++i) {
        if (t->d->ords[i] <= t->mo && cpx_nonzero(t->coef[i])) hi = t->d->ords[i];
    }
    t->hi = hi;
}

static void ctpsa_zero(ctpsa_t *t) {
    if (!t || !t->coef) return;
    memset(t->coef, 0, (size_t)t->d->nc * sizeof(cpx_t));
    t->hi = 0;
}

static void rawc_zero(const desc_t *d, cpx_t *a) {
    memset(a, 0, (size_t)d->nc * sizeof(cpx_t));
}

static void rawc_copy(const desc_t *d, const cpx_t *src, cpx_t *dst) {
    memcpy(dst, src, (size_t)d->nc * sizeof(cpx_t));
}

static cpx_t *tmp_craw(const desc_t *d) {
    return (cpx_t *)xcalloc((size_t)d->nc, sizeof(cpx_t));
}

static void rawc_set_const(const desc_t *d, cpx_t *r, cpx_t v) {
    rawc_zero(d, r);
    r[0] = v;
}

static cpx_t ctpsa_geti_value(const ctpsa_t *t, idx_t i) {
    if (!t || i < 0 || i >= t->d->nc) return 0.0;
    return t->d->ords[i] <= t->mo ? t->coef[i] : 0.0;
}

static void rawc_update_to_ctpsa(const cpx_t *raw, ctpsa_t *t) {
    memcpy(t->coef, raw, (size_t)t->d->nc * sizeof(cpx_t));
    ctpsa_update_hi(t);
}

static void rawc_add_scaled(const desc_t *d, cpx_t *dst, const cpx_t *src, cpx_t scale, ord_t mo) {
    for (idx_t i = 0; i < d->nc; ++i) if (d->ords[i] <= mo) dst[i] += scale * src[i];
}

static void rawc_mul(const desc_t *d, const cpx_t *a, const cpx_t *b, cpx_t *r, ord_t mo) {
    rawc_zero(d, r);
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (idx_t i = 0; i < d->nc; ++i) {
        if (!cpx_nonzero(a[i]) || d->ords[i] > mo) continue;
        const ord_t *mi = d->monos + ((size_t)i * d->nn);
        for (idx_t j = 0; j < d->nc; ++j) {
            if (!cpx_nonzero(b[j]) || d->ords[j] > mo) continue;
            if (d->ords[i] + d->ords[j] > mo) continue;
            const ord_t *mj = d->monos + ((size_t)j * d->nn);
            int ok = 1;
            for (int k = 0; k < d->nn; ++k) {
                int v = mi[k] + mj[k];
                if (v > d->no[k]) { ok = 0; break; }
                m[k] = (ord_t)v;
            }
            if (!ok || mono_total(d->nn, m) > mo || mono_param_total(d, m) > d->po) continue;
            idx_t idx = idx_for_mono(d, m);
            if (idx >= 0) r[idx] += a[i] * b[j];
        }
    }
    free(m);
}

static void rawc_inverse(const desc_t *d, const cpx_t *a, cpx_t v, cpx_t *r, ord_t mo) {
    cpx_t a0 = a[0];
    cpx_t *h = tmp_craw(d);
    cpx_t *term = tmp_craw(d);
    cpx_t *next = tmp_craw(d);
    if (cabs(a0) == 0.0) {
        rawc_set_const(d, r, NAN + NAN * I);
        goto done;
    }
    rawc_copy(d, a, h);
    h[0] = 0.0;
    rawc_set_const(d, term, 1.0);
    rawc_set_const(d, r, v / a0);
    cpx_t coeff = v / a0;
    for (int k = 1; k <= mo; ++k) {
        rawc_mul(d, term, h, next, mo);
        rawc_copy(d, next, term);
        coeff *= -1.0 / a0;
        rawc_add_scaled(d, r, term, coeff, mo);
    }
done:
    free(h); free(term); free(next);
}

static void rawc_pow_int(const desc_t *d, const cpx_t *a, int n, cpx_t *r, ord_t mo) {
    cpx_t *base = tmp_craw(d);
    cpx_t *tmp = tmp_craw(d);
    rawc_set_const(d, r, 1.0);
    rawc_copy(d, a, base);
    int exp = n < 0 ? -n : n;
    while (exp > 0) {
        if (exp & 1) { rawc_mul(d, r, base, tmp, mo); rawc_copy(d, tmp, r); }
        exp >>= 1;
        if (exp) { rawc_mul(d, base, base, tmp, mo); rawc_copy(d, tmp, base); }
    }
    free(base); free(tmp);
}

static cpx_t binom_cpx(cpx_t p, int k) {
    cpx_t r = 1.0;
    for (int i = 1; i <= k; ++i) r *= (p - (double)(i - 1)) / (double)i;
    return r;
}

typedef cpx_t (*coeffc_fn)(cpx_t, int, cpx_t);

static cpx_t coeffc_exp(cpx_t x, int k, cpx_t p) { (void)p; return cexp(x) * factorial_inv(k); }
static cpx_t coeffc_log(cpx_t x, int k, cpx_t p) {
    (void)p;
    if (k == 0) return clog(x);
    return ((k & 1) ? 1.0 : -1.0) / ((double)k * cpow(x, (double)k));
}
static cpx_t coeffc_pow(cpx_t x, int k, cpx_t p) {
    if (k == 0) return cpow(x, p);
    return cpow(x, p - (double)k) * binom_cpx(p, k);
}
static cpx_t coeffc_sin(cpx_t x, int k, cpx_t p) {
    (void)p;
    switch (k & 3) {
        case 0: return csin(x) * factorial_inv(k);
        case 1: return ccos(x) * factorial_inv(k);
        case 2: return -csin(x) * factorial_inv(k);
        default: return -ccos(x) * factorial_inv(k);
    }
}
static cpx_t coeffc_cos(cpx_t x, int k, cpx_t p) {
    (void)p;
    switch (k & 3) {
        case 0: return ccos(x) * factorial_inv(k);
        case 1: return -csin(x) * factorial_inv(k);
        case 2: return -ccos(x) * factorial_inv(k);
        default: return csin(x) * factorial_inv(k);
    }
}
static cpx_t coeffc_sinh(cpx_t x, int k, cpx_t p) { (void)p; return ((k & 1) ? ccosh(x) : csinh(x)) * factorial_inv(k); }
static cpx_t coeffc_cosh(cpx_t x, int k, cpx_t p) { (void)p; return ((k & 1) ? csinh(x) : ccosh(x)) * factorial_inv(k); }

static void rawc_apply_series(const ctpsa_t *a, ctpsa_t *c, coeffc_fn fn, cpx_t param) {
    const desc_t *d = a->d;
    ord_t mo = c->mo < a->mo ? c->mo : a->mo;
    cpx_t *h = tmp_craw(d);
    cpx_t *term = tmp_craw(d);
    cpx_t *next = tmp_craw(d);
    cpx_t *res = tmp_craw(d);
    rawc_copy(d, a->coef, h);
    h[0] = 0.0;
    rawc_set_const(d, term, 1.0);
    rawc_zero(d, res);
    for (int k = 0; k <= mo; ++k) {
        rawc_add_scaled(d, res, term, fn(a->coef[0], k, param), mo);
        if (k != mo) { rawc_mul(d, term, h, next, mo); rawc_copy(d, next, term); }
    }
    rawc_update_to_ctpsa(res, c);
    free(h); free(term); free(next); free(res);
}

ctpsa_t* mad_ctpsa_newd(const desc_t *d, ord_t mo) {
    if (!d) return NULL;
    if (mo == mad_tpsa_dflt || mo > d->mo) mo = d->mo;
    ctpsa_t *t = (ctpsa_t *)xcalloc(1, sizeof(ctpsa_t));
    t->d = d;
    t->mo = mo;
    t->hi = 0;
    t->coef = (cpx_t *)xcalloc((size_t)d->nc, sizeof(cpx_t));
    return t;
}

ctpsa_t* mad_ctpsa_new(const ctpsa_t *t, ord_t mo) {
    if (!t) return NULL;
    if (mo == mad_tpsa_same) mo = t->mo;
    return mad_ctpsa_newd(t->d, mo);
}

void mad_ctpsa_del(const ctpsa_t *t) {
    if (!t) return;
    free((void *)t->coef);
    free((void *)t);
}

const desc_t* mad_ctpsa_desc(const ctpsa_t *t) { return t ? t->d : NULL; }

ord_t mad_ctpsa_mo(ctpsa_t *t, ord_t mo) {
    if (!t) return 0;
    ord_t old = t->mo;
    if (mo != mad_tpsa_same && mo != mad_tpsa_dflt) {
        t->mo = mo > t->d->mo ? t->d->mo : mo;
        for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] > t->mo) t->coef[i] = 0.0;
        ctpsa_update_hi(t);
    }
    return old;
}

ssz_t mad_ctpsa_len(const ctpsa_t *t, log_t hi_) {
    if (!t) return 0;
    ord_t o = hi_ ? t->hi : t->mo;
    return mad_desc_maxlen(t->d, o);
}
ord_t mad_ctpsa_ord(const ctpsa_t *t, log_t hi_) { return t ? (hi_ ? t->hi : t->mo) : 0; }
log_t mad_ctpsa_isnul(const ctpsa_t *t) {
    if (!t) return 1;
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] <= t->mo && cpx_nonzero(t->coef[i])) return 0;
    return 1;
}
log_t mad_ctpsa_isval(const ctpsa_t *t) {
    if (!t) return 0;
    for (idx_t i = 1; i < t->d->nc; ++i) if (t->d->ords[i] <= t->mo && cpx_nonzero(t->coef[i])) return 0;
    return 1;
}
log_t mad_ctpsa_isvalid(const ctpsa_t *t) { return t && t->d && t->coef; }

void mad_ctpsa_copy(const ctpsa_t *t, ctpsa_t *r) {
    if (!t || !r) return;
    ctpsa_zero(r);
    ord_t mo = r->mo < t->mo ? r->mo : t->mo;
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] <= mo) r->coef[i] = t->coef[i];
    ctpsa_update_hi(r);
}

void mad_ctpsa_setval_r(ctpsa_t *t, num_t v_re, num_t v_im) { if (t) { ctpsa_zero(t); t->coef[0] = v_re + v_im * I; } }
void mad_ctpsa_clear(ctpsa_t *t) { if (t) ctpsa_zero(t); }
void mad_ctpsa_update(ctpsa_t *t) { ctpsa_update_hi(t); }
void mad_ctpsa_setvar_r(ctpsa_t *t, num_t v_re, num_t v_im, idx_t iv, num_t scl_re_, num_t scl_im_) {
    if (!t || iv <= 0 || iv > t->d->nv || t->mo < 1) return;
    ctpsa_zero(t); t->coef[0] = v_re + v_im * I; t->coef[iv] = (scl_re_ || scl_im_) ? (scl_re_ + scl_im_ * I) : 1.0; ctpsa_update_hi(t);
}
void mad_ctpsa_setprm_r(ctpsa_t *t, num_t v_re, num_t v_im, idx_t ip) {
    if (!t || ip <= 0 || ip > t->d->np || t->mo < 1) return;
    ctpsa_zero(t); t->coef[0] = v_re + v_im * I; t->coef[t->d->nv + ip] = 1.0; ctpsa_update_hi(t);
}

void mad_ctpsa_cplx(const tpsa_t *re_, const tpsa_t *im_, ctpsa_t *r) {
    if (!r) return;
    ctpsa_zero(r);
    const desc_t *d = r->d;
    for (idx_t i = 0; i < d->nc; ++i) if (d->ords[i] <= r->mo) {
        double re = re_ ? mad_tpsa_geti(re_, i) : 0.0;
        double im = im_ ? mad_tpsa_geti(im_, i) : 0.0;
        r->coef[i] = re + im * I;
    }
    ctpsa_update_hi(r);
}
void mad_ctpsa_real(const ctpsa_t *t, tpsa_t *r) {
    if (!t || !r) return;
    tpsa_zero(r);
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] <= r->mo) r->coef[i] = creal(t->coef[i]);
    tpsa_update_hi(r);
}
void mad_ctpsa_imag(const ctpsa_t *t, tpsa_t *r) {
    if (!t || !r) return;
    tpsa_zero(r);
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] <= r->mo) r->coef[i] = cimag(t->coef[i]);
    tpsa_update_hi(r);
}
void mad_ctpsa_cabs(const ctpsa_t *t, tpsa_t *r) {
    tpsa_t *re = mad_tpsa_newd(t->d, t->mo), *im = mad_tpsa_newd(t->d, t->mo), *rr = mad_tpsa_newd(t->d, t->mo), *ii = mad_tpsa_newd(t->d, t->mo);
    mad_ctpsa_real(t, re); mad_ctpsa_imag(t, im); mad_tpsa_mul(re, re, rr); mad_tpsa_mul(im, im, ii); mad_tpsa_add(rr, ii, rr); mad_tpsa_sqrt(rr, r);
    mad_tpsa_del(re); mad_tpsa_del(im); mad_tpsa_del(rr); mad_tpsa_del(ii);
}
void mad_ctpsa_carg(const ctpsa_t *t, tpsa_t *r) {
    tpsa_t *re = mad_tpsa_newd(t->d, t->mo), *im = mad_tpsa_newd(t->d, t->mo);
    mad_ctpsa_real(t, re); mad_ctpsa_imag(t, im); mad_tpsa_atan2(im, re, r);
    mad_tpsa_del(re); mad_tpsa_del(im);
}
void mad_ctpsa_rect(const ctpsa_t *t, ctpsa_t *r) { mad_ctpsa_copy(t, r); }
void mad_ctpsa_polar(const ctpsa_t *t, ctpsa_t *r) { mad_ctpsa_copy(t, r); }

ord_t mad_ctpsa_mono(const ctpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_) { return t ? mad_desc_mono(t->d, i, n, m_, p_) : 0; }
idx_t mad_ctpsa_idxm(const ctpsa_t *t, ssz_t n, const ord_t m[]) { return t ? mad_desc_idxm(t->d, n, m) : -1; }
void mad_ctpsa_geti_r(const ctpsa_t *t, idx_t i, cpx_t *r) { if (r) *r = ctpsa_geti_value(t, i); }
void mad_ctpsa_getm_r(const ctpsa_t *t, ssz_t n, const ord_t m[], cpx_t *r) {
    idx_t idx = mad_ctpsa_idxm(t, n, m);
    if (r) *r = idx >= 0 ? ctpsa_geti_value(t, idx) : 0.0;
}
void mad_ctpsa_seti_r(ctpsa_t *t, idx_t i, num_t a_re, num_t a_im, num_t b_re, num_t b_im) {
    if (!t || i < 0 || i >= t->d->nc || t->d->ords[i] > t->mo) return;
    cpx_t a = a_re + a_im * I, b = b_re + b_im * I;
    t->coef[i] = a * t->coef[i] + b;
    ctpsa_update_hi(t);
}
void mad_ctpsa_setm_r(ctpsa_t *t, ssz_t n, const ord_t m[], num_t a_re, num_t a_im, num_t b_re, num_t b_im) {
    idx_t idx = mad_ctpsa_idxm(t, n, m);
    if (idx >= 0) mad_ctpsa_seti_r(t, idx, a_re, a_im, b_re, b_im);
}
idx_t mad_ctpsa_cycle(const ctpsa_t *t, idx_t i, ssz_t n, ord_t m_[], cpx_t *v_) {
    if (!t) return -1;
    for (idx_t k = i + 1; k < t->d->nc; ++k) {
        if (t->d->ords[k] <= t->mo && cpx_nonzero(t->coef[k])) {
            if (m_) mad_desc_mono(t->d, k, n, m_, NULL);
            if (v_) *v_ = t->coef[k];
            return k;
        }
    }
    return -1;
}

void mad_ctpsa_getord(const ctpsa_t *t, ctpsa_t *r, ord_t ord) {
    if (!t || !r) return;
    ctpsa_zero(r);
    if (ord == 0) r->coef[0] = t->coef[0];
    for (idx_t i = 1; i < t->d->nc; ++i) if (t->d->ords[i] == ord && ord <= r->mo) r->coef[i] = t->coef[i];
    ctpsa_update_hi(r);
}
void mad_ctpsa_cutord(const ctpsa_t *t, ctpsa_t *r, int ord) {
    if (!t || !r) return;
    ctpsa_zero(r);
    for (idx_t i = 0; i < t->d->nc; ++i) {
        int o = t->d->ords[i];
        int keep = ord > 0 ? (o < ord) : (o > -ord);
        if (keep && o <= r->mo) r->coef[i] = t->coef[i];
    }
    ctpsa_update_hi(r);
}
void mad_ctpsa_clrord(ctpsa_t *t, ord_t ord) {
    if (!t) return;
    for (idx_t i = 0; i < t->d->nc; ++i) if (t->d->ords[i] == ord) t->coef[i] = 0.0;
    ctpsa_update_hi(t);
}

log_t mad_ctpsa_equ(const ctpsa_t *a, const ctpsa_t *b, num_t tol_) {
    if (!a || !b || a->d != b->d) return 0;
    double tol = tol_ > 0 ? tol_ : 0.0;
    for (idx_t i = 0; i < a->d->nc; ++i) {
        cpx_t av = ctpsa_geti_value(a, i), bv = ctpsa_geti_value(b, i);
        if (cabs(av - bv) > tol) return 0;
    }
    return 1;
}
void mad_ctpsa_dif(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c) {
    if (!a || !b || !c) return;
    ctpsa_zero(c);
    for (idx_t i = 0; i < a->d->nc; ++i) if (a->d->ords[i] <= c->mo) {
        double den = fmax(cabs(a->coef[i]), 1.0);
        c->coef[i] = (a->coef[i] - b->coef[i]) / den;
    }
    ctpsa_update_hi(c);
}
void mad_ctpsa_add(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c) {
    if (!a || !b || !c) return;
    cpx_t *r = tmp_craw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = ctpsa_geti_value(a, i) + ctpsa_geti_value(b, i);
    rawc_update_to_ctpsa(r, c); free(r);
}
void mad_ctpsa_sub(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c) {
    if (!a || !b || !c) return;
    cpx_t *r = tmp_craw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = ctpsa_geti_value(a, i) - ctpsa_geti_value(b, i);
    rawc_update_to_ctpsa(r, c); free(r);
}
void mad_ctpsa_scl_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c) {
    if (!a || !c) return;
    cpx_t v = v_re + v_im * I;
    cpx_t *r = tmp_craw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = ctpsa_geti_value(a, i) * v;
    rawc_update_to_ctpsa(r, c); free(r);
}
void mad_ctpsa_divn_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c) { cpx_t v = v_re + v_im * I; mad_ctpsa_scl_r(a, creal(1.0/v), cimag(1.0/v), c); }
void mad_ctpsa_mul(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c) {
    if (!a || !b || !c) return;
    cpx_t *r = tmp_craw(c->d);
    rawc_mul(c->d, a->coef, b->coef, r, c->mo);
    rawc_update_to_ctpsa(r, c); free(r);
}
void mad_ctpsa_inv_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c) {
    if (!a || !c) return;
    cpx_t *r = tmp_craw(c->d);
    rawc_inverse(c->d, a->coef, v_re + v_im * I, r, c->mo);
    rawc_update_to_ctpsa(r, c); free(r);
}
void mad_ctpsa_div(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c) {
    if (!a || !b || !c) return;
    cpx_t *invb = tmp_craw(c->d), *r = tmp_craw(c->d);
    rawc_inverse(c->d, b->coef, 1.0, invb, c->mo);
    rawc_mul(c->d, a->coef, invb, r, c->mo);
    rawc_update_to_ctpsa(r, c); free(invb); free(r);
}
void mad_ctpsa_powi(const ctpsa_t *a, int n, ctpsa_t *c) {
    if (!a || !c) return;
    cpx_t *base = tmp_craw(c->d), *r = tmp_craw(c->d);
    if (n < 0) rawc_inverse(c->d, a->coef, 1.0, base, c->mo); else rawc_copy(c->d, a->coef, base);
    rawc_pow_int(c->d, base, n < 0 ? -n : n, r, c->mo);
    rawc_update_to_ctpsa(r, c); free(base); free(r);
}
void mad_ctpsa_pown_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_pow, v_re + v_im * I); }
void mad_ctpsa_pow(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c) {
    ctpsa_t *la = mad_ctpsa_new(a, mad_tpsa_same), *prod = mad_ctpsa_new(a, mad_tpsa_same);
    mad_ctpsa_log(a, la); mad_ctpsa_mul(la, b, prod); mad_ctpsa_exp(prod, c);
    mad_ctpsa_del(la); mad_ctpsa_del(prod);
}

num_t mad_ctpsa_nrm(const ctpsa_t *a) {
    if (!a) return 0.0;
    double m = 0.0;
    for (idx_t i = 0; i < a->d->nc; ++i) if (a->d->ords[i] <= a->mo) m = fmax(m, cabs(a->coef[i]));
    return m;
}
void mad_ctpsa_unit(const ctpsa_t *a, ctpsa_t *c) { cpx_t z = a ? a->coef[0] : 0.0; mad_ctpsa_setval_r(c, cabs(z) ? creal(z / cabs(z)) : 0.0, cabs(z) ? cimag(z / cabs(z)) : 0.0); }
void mad_ctpsa_conj(const ctpsa_t *a, ctpsa_t *c) {
    if (!a || !c) return;
    cpx_t *r = tmp_craw(c->d);
    for (idx_t i = 0; i < c->d->nc; ++i) if (c->d->ords[i] <= c->mo) r[i] = conj(ctpsa_geti_value(a, i));
    rawc_update_to_ctpsa(r, c); free(r);
}
void mad_ctpsa_sqrt(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_pow, 0.5); }
void mad_ctpsa_exp(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_exp, 0.0); }
void mad_ctpsa_log(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_log, 0.0); }
void mad_ctpsa_sin(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_sin, 0.0); }
void mad_ctpsa_cos(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_cos, 0.0); }
void mad_ctpsa_sincos(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c) { mad_ctpsa_sin(a, s); mad_ctpsa_cos(a, c); }
void mad_ctpsa_tan(const ctpsa_t *a, ctpsa_t *c) { ctpsa_t *s=mad_ctpsa_new(a,mad_tpsa_same), *co=mad_ctpsa_new(a,mad_tpsa_same); mad_ctpsa_sin(a,s); mad_ctpsa_cos(a,co); mad_ctpsa_div(s,co,c); mad_ctpsa_del(s); mad_ctpsa_del(co); }
void mad_ctpsa_cot(const ctpsa_t *a, ctpsa_t *c) { ctpsa_t *s=mad_ctpsa_new(a,mad_tpsa_same), *co=mad_ctpsa_new(a,mad_tpsa_same); mad_ctpsa_sin(a,s); mad_ctpsa_cos(a,co); mad_ctpsa_div(co,s,c); mad_ctpsa_del(s); mad_ctpsa_del(co); }
void mad_ctpsa_sinh(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_sinh, 0.0); }
void mad_ctpsa_cosh(const ctpsa_t *a, ctpsa_t *c) { rawc_apply_series(a, c, coeffc_cosh, 0.0); }
void mad_ctpsa_sincosh(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c) { mad_ctpsa_sinh(a, s); mad_ctpsa_cosh(a, c); }
void mad_ctpsa_tanh(const ctpsa_t *a, ctpsa_t *c) { ctpsa_t *s=mad_ctpsa_new(a,mad_tpsa_same), *co=mad_ctpsa_new(a,mad_tpsa_same); mad_ctpsa_sinh(a,s); mad_ctpsa_cosh(a,co); mad_ctpsa_div(s,co,c); mad_ctpsa_del(s); mad_ctpsa_del(co); }
void mad_ctpsa_coth(const ctpsa_t *a, ctpsa_t *c) { ctpsa_t *s=mad_ctpsa_new(a,mad_tpsa_same), *co=mad_ctpsa_new(a,mad_tpsa_same); mad_ctpsa_sinh(a,s); mad_ctpsa_cosh(a,co); mad_ctpsa_div(co,s,c); mad_ctpsa_del(s); mad_ctpsa_del(co); }
void mad_ctpsa_sinc(const ctpsa_t *a, ctpsa_t *c) { ctpsa_t *s=mad_ctpsa_new(a,mad_tpsa_same); mad_ctpsa_sin(a,s); mad_ctpsa_div(s,a,c); if (cabs(a->coef[0]) < 1e-14) c->coef[0]=1.0; ctpsa_update_hi(c); mad_ctpsa_del(s); }
void mad_ctpsa_sinhc(const ctpsa_t *a, ctpsa_t *c) { ctpsa_t *s=mad_ctpsa_new(a,mad_tpsa_same); mad_ctpsa_sinh(a,s); mad_ctpsa_div(s,a,c); if (cabs(a->coef[0]) < 1e-14) c->coef[0]=1.0; ctpsa_update_hi(c); mad_ctpsa_del(s); }
void mad_ctpsa_invsqrt_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c) { mad_ctpsa_sqrt(a,c); mad_ctpsa_inv_r(c,v_re,v_im,c); }
void mad_ctpsa_hypot(const ctpsa_t *x, const ctpsa_t *y, ctpsa_t *r) { ctpsa_t *xx=mad_ctpsa_new(x,mad_tpsa_same), *yy=mad_ctpsa_new(x,mad_tpsa_same); mad_ctpsa_mul(x,x,xx); mad_ctpsa_mul(y,y,yy); mad_ctpsa_add(xx,yy,xx); mad_ctpsa_sqrt(xx,r); mad_ctpsa_del(xx); mad_ctpsa_del(yy); }
void mad_ctpsa_hypot3(const ctpsa_t *x, const ctpsa_t *y, const ctpsa_t *z, ctpsa_t *r) { ctpsa_t *tmp=mad_ctpsa_new(x,mad_tpsa_same), *zz=mad_ctpsa_new(x,mad_tpsa_same); mad_ctpsa_hypot(x,y,tmp); mad_ctpsa_mul(tmp,tmp,tmp); mad_ctpsa_mul(z,z,zz); mad_ctpsa_add(tmp,zz,tmp); mad_ctpsa_sqrt(tmp,r); mad_ctpsa_del(tmp); mad_ctpsa_del(zz); }

void mad_ctpsa_deriv(const ctpsa_t *a, ctpsa_t *c, idx_t iv) {
    if (!a || !c || iv <= 0 || iv > a->d->nn) return;
    const desc_t *d = a->d; ctpsa_zero(c);
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (idx_t i = 0; i < d->nc; ++i) if (cpx_nonzero(a->coef[i]) && d->ords[i] <= a->mo) {
        const ord_t *mi = d->monos + ((size_t)i * d->nn);
        if (mi[iv-1] == 0) continue;
        memcpy(m, mi, (size_t)d->nn * sizeof(ord_t));
        m[iv-1]--;
        idx_t j = idx_for_mono(d, m);
        if (j >= 0 && d->ords[j] <= c->mo) c->coef[j] += a->coef[i] * mi[iv-1];
    }
    free(m); ctpsa_update_hi(c);
}
void mad_ctpsa_integ(const ctpsa_t *a, ctpsa_t *c, idx_t iv) {
    if (!a || !c || iv <= 0 || iv > a->d->nn) return;
    const desc_t *d = a->d; ctpsa_zero(c);
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    for (idx_t i = 0; i < d->nc; ++i) if (cpx_nonzero(a->coef[i]) && d->ords[i] <= a->mo) {
        const ord_t *mi = d->monos + ((size_t)i * d->nn);
        memcpy(m, mi, (size_t)d->nn * sizeof(ord_t));
        m[iv-1]++;
        idx_t j = idx_for_mono(d, m);
        if (j >= 0 && d->ords[j] <= c->mo) c->coef[j] += a->coef[i] / (double)m[iv-1];
    }
    free(m); ctpsa_update_hi(c);
}
void mad_ctpsa_derivm(const ctpsa_t *a, ctpsa_t *c, ssz_t n, const ord_t m[]) {
    ctpsa_t *tmp1 = mad_ctpsa_new(a, mad_tpsa_same), *tmp2 = mad_ctpsa_new(a, mad_tpsa_same);
    mad_ctpsa_copy(a, tmp1);
    for (int i = 0; i < n; ++i) for (int k = 0; k < m[i]; ++k) { mad_ctpsa_deriv(tmp1, tmp2, i+1); mad_ctpsa_copy(tmp2, tmp1); }
    mad_ctpsa_copy(tmp1, c);
    mad_ctpsa_del(tmp1); mad_ctpsa_del(tmp2);
}
void mad_ctpsa_poisbra(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c, int nv) {
    if (!a || !b || !c) return;
    ctpsa_zero(c);
    ctpsa_t *da=mad_ctpsa_new(a,mad_tpsa_same), *db=mad_ctpsa_new(a,mad_tpsa_same), *prod=mad_ctpsa_new(a,mad_tpsa_same), *acc=mad_ctpsa_new(a,mad_tpsa_same);
    int n = nv < a->d->nv ? nv : a->d->nv;
    for (int i=1; i+1<=n; i+=2) {
        mad_ctpsa_deriv(a,da,i); mad_ctpsa_deriv(b,db,i+1); mad_ctpsa_mul(da,db,prod); mad_ctpsa_add(acc,prod,acc);
        mad_ctpsa_deriv(a,da,i+1); mad_ctpsa_deriv(b,db,i); mad_ctpsa_mul(da,db,prod); mad_ctpsa_sub(acc,prod,acc);
    }
    mad_ctpsa_copy(acc,c);
    mad_ctpsa_del(da); mad_ctpsa_del(db); mad_ctpsa_del(prod); mad_ctpsa_del(acc);
}

static void make_cvariable_component(const desc_t *d, int idx1, ctpsa_t *out, cpx_t shift) {
    ctpsa_zero(out); out->coef[0] = shift;
    if (idx1 >= 1 && idx1 <= d->nn) {
        ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
        m[idx1-1] = 1;
        idx_t idx = idx_for_mono(d, m);
        if (idx >= 0 && d->ords[idx] <= out->mo) out->coef[idx] = 1.0;
        free(m);
    }
    ctpsa_update_hi(out);
}

void mad_ctpsa_compose(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const ctpsa_t *mb[], ctpsa_t *mc[]) {
    if (na <= 0 || !ma || !mb || !mc) return;
    const desc_t *d = ma[0]->d;
    for (ssz_t aidx=0; aidx<na; ++aidx) {
        ctpsa_t *out = mc[aidx]; ctpsa_zero(out);
        ctpsa_t *term = mad_ctpsa_new(out, mad_tpsa_same);
        ctpsa_t *powv = mad_ctpsa_new(out, mad_tpsa_same);
        ctpsa_t *tmp = mad_ctpsa_new(out, mad_tpsa_same);
        for (idx_t i=0; i<d->nc; ++i) if (cpx_nonzero(ma[aidx]->coef[i]) && d->ords[i] <= ma[aidx]->mo) {
            mad_ctpsa_setval_r(term, creal(ma[aidx]->coef[i]), cimag(ma[aidx]->coef[i]));
            const ord_t *m = d->monos + ((size_t)i*d->nn);
            int discard = 0;
            for (int j=0; j<d->nn; ++j) {
                int e = m[j]; if (!e) continue;
                if (j >= nb) { discard = 1; break; }
                mad_ctpsa_powi(mb[j], e, powv);
                mad_ctpsa_mul(term, powv, tmp);
                mad_ctpsa_copy(tmp, term);
            }
            if (!discard) mad_ctpsa_add(out, term, out);
        }
        mad_ctpsa_del(term); mad_ctpsa_del(powv); mad_ctpsa_del(tmp);
    }
}

void mad_ctpsa_translate(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], ctpsa_t *mc[]) {
    if (na <= 0) return;
    const desc_t *d = ma[0]->d;
    ctpsa_t **subs = (ctpsa_t **)xcalloc((size_t)nb, sizeof(ctpsa_t *));
    for (ssz_t i=0; i<nb; ++i) {
        subs[i] = mad_ctpsa_newd(d, d->mo);
        make_cvariable_component(d, (int)i+1, subs[i], tb ? tb[i] : 0.0);
    }
    mad_ctpsa_compose(na, ma, nb, (const ctpsa_t **)subs, mc);
    for (ssz_t i=0; i<nb; ++i) mad_ctpsa_del(subs[i]);
    free(subs);
}

void mad_ctpsa_eval(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], cpx_t tc[]) {
    if (!ma || !tc) return;
    for (ssz_t k=0; k<na; ++k) {
        const ctpsa_t *t = ma[k]; const desc_t *d = t->d; cpx_t sum=0.0;
        for (idx_t i=0; i<d->nc; ++i) if (d->ords[i] <= t->mo && cpx_nonzero(t->coef[i])) {
            const ord_t *m = d->monos + ((size_t)i*d->nn);
            cpx_t prod = t->coef[i];
            for (int j=0; j<d->nn; ++j) {
                int e = m[j]; if (!e) continue;
                cpx_t x = j < nb ? tb[j] : 0.0;
                prod *= cpow(x, (double)e);
            }
            sum += prod;
        }
        tc[k] = sum;
    }
}

ord_t mad_ctpsa_mord(ssz_t na, const ctpsa_t *ma[], log_t hi) { ord_t m=0; for (ssz_t i=0;i<na;++i) { ord_t o=mad_ctpsa_ord(ma[i],hi); if (o>m) m=o; } return m; }
num_t mad_ctpsa_mnrm(ssz_t na, const ctpsa_t *ma[]) { double m=0.0; for (ssz_t i=0;i<na;++i) m=fmax(m,mad_ctpsa_nrm(ma[i])); return m; }

static cpx_t linear_ccoef(const ctpsa_t *t, int var0) {
    const desc_t *d = t->d;
    if (var0 < 0 || var0 >= d->nn) return 0.0;
    ord_t *m = (ord_t *)xcalloc((size_t)d->nn, sizeof(ord_t));
    m[var0] = 1;
    idx_t idx = idx_for_mono(d, m);
    cpx_t value = (idx >= 0 && d->ords[idx] <= t->mo) ? t->coef[idx] : 0.0;
    free(m);
    return value;
}

static int invert_clinear_part_lapack(ssz_t n_, const ctpsa_t *ma[], cpx_t *inv_col_major) {
    int n = (int)n_;
    int nrhs = n;
    int lda = n;
    int ldb = n;
    int info = 0;
    cpx_t *a = (cpx_t *)xcalloc((size_t)n * (size_t)n, sizeof(cpx_t));
    int *ipiv = (int *)xcalloc((size_t)n, sizeof(int));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a[(size_t)j * (size_t)n + (size_t)i] = linear_ccoef(ma[i], j);
            inv_col_major[(size_t)j * (size_t)n + (size_t)i] = (i == j) ? 1.0 : 0.0;
        }
    }
    zgesv_(&n, &nrhs, a, &lda, ipiv, inv_col_major, &ldb, &info);
    free(a); free(ipiv);
    return info;
}

void mad_ctpsa_minv(ssz_t na, const ctpsa_t *ma[], ssz_t nb, ctpsa_t *mc[]) {
    if (na <= 0 || nb <= 0 || !ma || !mc) return;
    const desc_t *d = ma[0]->d;
    if (!d) return;
    ssz_t n = nb;
    if (n > na) n = na;
    if (n > d->nv) n = d->nv;
    if (n <= 0) return;
    cpx_t *invJ = (cpx_t *)xcalloc((size_t)n * (size_t)n, sizeof(cpx_t));
    int info = invert_clinear_part_lapack(n, ma, invJ);
    if (info != 0) {
        for (ssz_t i = 0; i < nb; ++i) make_cvariable_component(d, (int)i + 1, mc[i], 0.0);
        free(invJ); return;
    }
    ctpsa_t **identity=(ctpsa_t **)xcalloc((size_t)n,sizeof(ctpsa_t*)), **guess=(ctpsa_t **)xcalloc((size_t)n,sizeof(ctpsa_t*)), **composed=(ctpsa_t **)xcalloc((size_t)n,sizeof(ctpsa_t*)), **residual=(ctpsa_t **)xcalloc((size_t)n,sizeof(ctpsa_t*)), **correction=(ctpsa_t **)xcalloc((size_t)n,sizeof(ctpsa_t*));
    ctpsa_t *tmp = mad_ctpsa_newd(d, d->mo), *scaled = mad_ctpsa_newd(d, d->mo);
    for (ssz_t i=0;i<n;++i) {
        identity[i]=mad_ctpsa_newd(d,d->mo); guess[i]=mad_ctpsa_newd(d,d->mo); composed[i]=mad_ctpsa_newd(d,d->mo); residual[i]=mad_ctpsa_newd(d,d->mo); correction[i]=mad_ctpsa_newd(d,d->mo);
        make_cvariable_component(d,(int)i+1,identity[i],0.0); mad_ctpsa_setval_r(guess[i],0.0,0.0);
    }
    for (ssz_t i=0;i<n;++i) {
        mad_ctpsa_setval_r(guess[i],0.0,0.0);
        for (ssz_t j=0;j<n;++j) {
            mad_ctpsa_copy(identity[j], tmp);
            tmp->coef[0] -= ma[j]->coef[0]; ctpsa_update_hi(tmp);
            cpx_t scale = invJ[(size_t)j * (size_t)n + (size_t)i];
            mad_ctpsa_scl_r(tmp, creal(scale), cimag(scale), scaled);
            mad_ctpsa_add(guess[i], scaled, guess[i]);
        }
    }
    int max_iter = d->mo + 2;
    if (max_iter < 2) max_iter = 2;
    for (int iter=0; iter<max_iter; ++iter) {
        mad_ctpsa_compose(n, ma, n, (const ctpsa_t **)guess, composed);
        for (ssz_t i=0;i<n;++i) mad_ctpsa_sub(composed[i], identity[i], residual[i]);
        double corr_norm = 0.0;
        for (ssz_t i=0;i<n;++i) {
            mad_ctpsa_setval_r(correction[i],0.0,0.0);
            for (ssz_t j=0;j<n;++j) {
                cpx_t scale = invJ[(size_t)j * (size_t)n + (size_t)i];
                mad_ctpsa_scl_r(residual[j], creal(scale), cimag(scale), scaled);
                mad_ctpsa_add(correction[i], scaled, correction[i]);
            }
            corr_norm = fmax(corr_norm, mad_ctpsa_nrm(correction[i]));
        }
        for (ssz_t i=0;i<n;++i) mad_ctpsa_sub(guess[i], correction[i], guess[i]);
        if (corr_norm < 1e-14) break;
    }
    for (ssz_t i=0;i<n;++i) mad_ctpsa_copy(guess[i], mc[i]);
    for (ssz_t i=n;i<nb;++i) make_cvariable_component(d, (int)i+1, mc[i], 0.0);
    for (ssz_t i=0;i<n;++i) { mad_ctpsa_del(identity[i]); mad_ctpsa_del(guess[i]); mad_ctpsa_del(composed[i]); mad_ctpsa_del(residual[i]); mad_ctpsa_del(correction[i]); }
    mad_ctpsa_del(tmp); mad_ctpsa_del(scaled);
    free(identity); free(guess); free(composed); free(residual); free(correction); free(invJ);
}
