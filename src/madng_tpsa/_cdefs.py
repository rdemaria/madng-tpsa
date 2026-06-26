"""CFFI declarations shared by the ABI loader and bundled extension builder."""

CDEF = r"""
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
typedef double _Complex cpx_t;
typedef struct ctpsa_ ctpsa_t;

ctpsa_t*      mad_ctpsa_newd(const desc_t *d, ord_t mo);
ctpsa_t*      mad_ctpsa_new(const ctpsa_t *t, ord_t mo);
void          mad_ctpsa_del(const ctpsa_t *t);
const desc_t* mad_ctpsa_desc(const ctpsa_t *t);
ord_t         mad_ctpsa_mo(ctpsa_t *t, ord_t mo);
ssz_t         mad_ctpsa_len(const ctpsa_t *t, log_t hi_);
ord_t         mad_ctpsa_ord(const ctpsa_t *t, log_t hi_);
log_t         mad_ctpsa_isnul(const ctpsa_t *t);
log_t         mad_ctpsa_isval(const ctpsa_t *t);
log_t         mad_ctpsa_isvalid(const ctpsa_t *t);

void          mad_ctpsa_copy(const ctpsa_t *t, ctpsa_t *r);
void          mad_ctpsa_getord(const ctpsa_t *t, ctpsa_t *r, ord_t ord);
void          mad_ctpsa_cutord(const ctpsa_t *t, ctpsa_t *r, int ord);
void          mad_ctpsa_clrord(ctpsa_t *t, ord_t ord);
void          mad_ctpsa_setvar_r(ctpsa_t *t, num_t v_re, num_t v_im, idx_t iv, num_t scl_re_, num_t scl_im_);
void          mad_ctpsa_setprm_r(ctpsa_t *t, num_t v_re, num_t v_im, idx_t ip);
void          mad_ctpsa_setval_r(ctpsa_t *t, num_t v_re, num_t v_im);
void          mad_ctpsa_update(ctpsa_t *t);
void          mad_ctpsa_clear(ctpsa_t *t);

void          mad_ctpsa_cplx(const tpsa_t *re_, const tpsa_t *im_, ctpsa_t *r);
void          mad_ctpsa_real(const ctpsa_t *t, tpsa_t *r);
void          mad_ctpsa_imag(const ctpsa_t *t, tpsa_t *r);
void          mad_ctpsa_cabs(const ctpsa_t *t, tpsa_t *r);
void          mad_ctpsa_carg(const ctpsa_t *t, tpsa_t *r);
void          mad_ctpsa_rect(const ctpsa_t *t, ctpsa_t *r);
void          mad_ctpsa_polar(const ctpsa_t *t, ctpsa_t *r);

ord_t         mad_ctpsa_mono(const ctpsa_t *t, idx_t i, ssz_t n, ord_t m_[], ord_t *p_);
idx_t         mad_ctpsa_idxm(const ctpsa_t *t, ssz_t n, const ord_t m[]);
idx_t         mad_ctpsa_cycle(const ctpsa_t *t, idx_t i, ssz_t n, ord_t m_[], cpx_t *v_);
void          mad_ctpsa_geti_r(const ctpsa_t *t, idx_t i, cpx_t *r);
void          mad_ctpsa_getm_r(const ctpsa_t *t, ssz_t n, const ord_t m[], cpx_t *r);
void          mad_ctpsa_seti_r(ctpsa_t *t, idx_t i, num_t a_re, num_t a_im, num_t b_re, num_t b_im);
void          mad_ctpsa_setm_r(ctpsa_t *t, ssz_t n, const ord_t m[], num_t a_re, num_t a_im, num_t b_re, num_t b_im);

log_t         mad_ctpsa_equ(const ctpsa_t *a, const ctpsa_t *b, num_t tol_);
void          mad_ctpsa_dif(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c);
void          mad_ctpsa_add(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c);
void          mad_ctpsa_sub(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c);
void          mad_ctpsa_mul(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c);
void          mad_ctpsa_div(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c);
void          mad_ctpsa_pow(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c);
void          mad_ctpsa_powi(const ctpsa_t *a, int n, ctpsa_t *c);
void          mad_ctpsa_pown_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c);

num_t         mad_ctpsa_nrm(const ctpsa_t *a);
void          mad_ctpsa_unit(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_conj(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_sqrt(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_exp(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_log(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_sincos(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c);
void          mad_ctpsa_sin(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_cos(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_tan(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_cot(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_sinc(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_sincosh(const ctpsa_t *a, ctpsa_t *s, ctpsa_t *c);
void          mad_ctpsa_sinh(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_cosh(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_tanh(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_coth(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_sinhc(const ctpsa_t *a, ctpsa_t *c);
void          mad_ctpsa_scl_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c);
void          mad_ctpsa_divn_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c);
void          mad_ctpsa_inv_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c);
void          mad_ctpsa_invsqrt_r(const ctpsa_t *a, num_t v_re, num_t v_im, ctpsa_t *c);
void          mad_ctpsa_hypot(const ctpsa_t *x, const ctpsa_t *y, ctpsa_t *r);
void          mad_ctpsa_hypot3(const ctpsa_t *x, const ctpsa_t *y, const ctpsa_t *z, ctpsa_t *r);

void          mad_ctpsa_integ(const ctpsa_t *a, ctpsa_t *c, idx_t iv);
void          mad_ctpsa_deriv(const ctpsa_t *a, ctpsa_t *c, idx_t iv);
void          mad_ctpsa_derivm(const ctpsa_t *a, ctpsa_t *c, ssz_t n, const ord_t m[]);
void          mad_ctpsa_poisbra(const ctpsa_t *a, const ctpsa_t *b, ctpsa_t *c, int nv);

ord_t         mad_ctpsa_mord(ssz_t na, const ctpsa_t *ma[], log_t hi);
num_t         mad_ctpsa_mnrm(ssz_t na, const ctpsa_t *ma[]);
void          mad_ctpsa_minv(ssz_t na, const ctpsa_t *ma[], ssz_t nb, ctpsa_t *mc[]);
void          mad_ctpsa_compose(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const ctpsa_t *mb[], ctpsa_t *mc[]);
void          mad_ctpsa_translate(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], ctpsa_t *mc[]);
void          mad_ctpsa_eval(ssz_t na, const ctpsa_t *ma[], ssz_t nb, const cpx_t tb[], cpx_t tc[]);

"""
