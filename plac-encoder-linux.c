#include <alsa/asoundlib.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <sys/stat.h>

#define MAGIC16(a, b)                   ((u16)(a) | ((u16)(b) << 8))
#define MAGIC32(a, b, c, d)             ((u32)(a) | ((u32)(b) << 8) | ((u32)(c) << 16) | ((u32)(d) << 24))
#define MAGIC64(a, b, c, d, e, f, g, h) (MAGIC32(a, b, c, d) | ((u64)MAGIC32(e, f, g, h) << 32))

// ----------------------------------------------------------------------------------------------------
//; 类型定义

typedef int8_t          i8;
typedef uint8_t         u8;
typedef int16_t         i16;
typedef uint16_t        u16;
typedef int32_t         i32;
typedef uint32_t        u32;
typedef int64_t         i64;
typedef uint64_t        u64;
typedef float           f32;
typedef double          f64;
typedef _Complex float  cf32;
typedef _Complex double cf64;

#define bool  _Bool
#define true  ((bool)1)
#define false ((bool)0)

// ----------------------------------------------------------------------------------------------------
//; 文件写入

static int write_to_file(const char *filename, const void *data, size_t size) {
  if (filename == NULL || data == NULL) return -1;

  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;

  int nwrite = write(fd, data, size);
  if (nwrite != size) {
    close(fd);
    remove(filename);
    return -1;
  }

  close(fd);
  return 0;
}

// ----------------------------------------------------------------------------------------------------
//; 内存流

typedef struct mostream {
  void  *buf;
  size_t size;
  size_t capacity;
} *mostream_t;

static mostream_t mostream_alloc(size_t capacity) {
  mostream_t stream = malloc(sizeof(struct mostream));
  if (stream == NULL) return NULL;
  stream->buf      = malloc(capacity);
  stream->size     = 0;
  stream->capacity = capacity;
  return stream;
}

static void mostream_free(mostream_t stream) {
  if (stream == NULL) return;
  free(stream->buf);
  free(stream);
}

static size_t mostream_write(mostream_t stream, const void *data, size_t size) {
  if (stream->size + size > stream->capacity) {
    size_t new_capacity = stream->capacity * 2;
    while (new_capacity < stream->size + size) {
      new_capacity *= 2;
    }
    char *new_buffer = realloc(stream->buf, new_capacity);
    if (new_buffer == NULL) return 0;
    stream->buf      = new_buffer;
    stream->capacity = new_capacity;
  }
  memcpy(stream->buf + stream->size, data, size);
  stream->size += size;
  return size;
}

typedef struct mobitstream {
  void  *buf;
  size_t size;
  size_t capacity;
  size_t bit_pos;
} *mobitstream_t;

static mobitstream_t mobitstream_alloc(size_t capacity) {
  mobitstream_t stream = malloc(sizeof(struct mobitstream));
  if (stream == NULL) return NULL;
  stream->buf      = malloc(capacity);
  stream->size     = 0;
  stream->capacity = capacity;
  stream->bit_pos  = 0;
  return stream;
}

static void mobitstream_free(mobitstream_t stream) {
  if (stream == NULL) return;
  free(stream->buf);
  free(stream);
}

static size_t mobitstream_write_bit(mobitstream_t stream, bool bit) {
  if (stream->size * 8 + stream->bit_pos >= stream->capacity * 8) {
    size_t new_capacity = stream->capacity * 2;
    while (new_capacity * 8 < stream->size * 8 + stream->bit_pos + 1) {
      new_capacity *= 2;
    }
    uint8_t *new_buffer = realloc(stream->buf, new_capacity);
    if (new_buffer == NULL) return 0;
    stream->buf      = new_buffer;
    stream->capacity = new_capacity;
  }
  if (bit) {
    ((uint8_t *)stream->buf)[stream->size] |= (1 << (7 - stream->bit_pos));
  } else {
    ((uint8_t *)stream->buf)[stream->size] &= ~(1 << (7 - stream->bit_pos));
  }
  stream->bit_pos++;
  if (stream->bit_pos == 8) {
    stream->bit_pos = 0;
    stream->size++;
  }
  return 1;
}

static size_t mobitstream_write_bits(mobitstream_t stream, size_t bits, size_t nbits) {
  size_t written = 0;
  for (size_t i = 0; i < nbits; i++) {
    size_t result = mobitstream_write_bit(stream, (bits >> i) & 1);
    if (result == 0) return written;
    written++;
  }
  return written;
}

// ----------------------------------------------------------------------------------------------------
//; volume_fine_tuning

static void _volume_fine_tuning2(f32 *data, size_t len, size_t k) {
  for (size_t i = k; i < len - k; i++) {
    bool s = data[i] < 0;
    f32  x = __builtin_fabsf(data[i]);
    f32  p = __builtin_fabsf(data[i - k]);
    f32  n = __builtin_fabsf(data[i + k]);
    if (p != 0 && x > p * 4) {
      x           += p;
      data[i - k]  = 0;
    }
    if (n != 0 && x > n * 4) {
      x           += n;
      data[i + k]  = 0;
    }
    data[i] = s ? -x : x;
  }
}

static void _volume_fine_tuning1(f32 *data, size_t len, f32 vol) {
  for (size_t i = 0; i < len; i++) {
    f32 x = __builtin_fabsf(data[i]);
    if (x * 96 < vol) data[i] = 0;
  }
}

static void volume_fine_tuning(f32 *data, size_t len) {
  f32 max = 0;
  for (size_t i = 0; i < len; i++) {
    f32 x = __builtin_fabsf(data[i]);
    if (x > max) max = x;
  }
  _volume_fine_tuning1(data, len, max);
  _volume_fine_tuning2(data, len, 1);
  _volume_fine_tuning2(data, len, 2);
}

// ----------------------------------------------------------------------------------------------------
//; mulaw

#define MU 1023

static void mulaw_compress(f32 *data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    f32  x    = data[i];
    bool sign = x < 0;
    if (sign) x = -x;
    x       = __builtin_log(1 + MU * x) / __builtin_log(1 + MU);
    data[i] = sign ? -x : x;
  }
}

#undef MU

// ----------------------------------------------------------------------------------------------------
//; 量化

typedef struct quantized {
  i16    max;
  i16    min;
  i16    mid;
  i16    nbit;
  size_t len;
  f32   *dataf;
  i16   *datai;
} *quantized_t;

static void quantize(quantized_t q) {
  if (q->nbit == 0) goto zero;
  f32 max = q->dataf[0];
  f32 min = q->dataf[0];
  for (size_t i = 1; i < q->len; i++) {
    if (q->dataf[i] > max) max = q->dataf[i];
    if (q->dataf[i] < min) min = q->dataf[i];
  }
  if (-1e-6 < max - min && max - min < 1e-6) goto zero;
  q->max      = __builtin_ceil(max * 256);
  q->min      = __builtin_floor(min * 256);
  max         = (f32)q->max / 256;
  min         = (f32)q->min / 256;
  const f32 k = ((1 << q->nbit) - 1) / (max - min);
  q->mid      = (i16)(-min * k);
  for (size_t i = 0; i < q->len; i++) {
    q->datai[i] = (i16)((q->dataf[i] - min) * k);
  }
  return;

zero:
  q->mid = 0;
  for (size_t i = 0; i < q->len; i++) {
    q->datai[i] = 0;
  }
}

static f32 quantize_diff(quantized_t q) {
  f32 max = (f32)q->max / 256;
  f32 min = (f32)q->min / 256;
  if (q->nbit == 0) {
    f32 diff = 0;
    for (size_t i = 0; i < q->len; i++) {
      diff += (q->dataf[i] - min) * (q->dataf[i] - min);
    }
    return diff / q->len;
  }
  f32       diff = 0;
  const f32 k    = (max - min) / ((1 << q->nbit) - 1);
  for (size_t i = 0; i < q->len; i++) {
    f32 d  = q->datai[i] * k + min;
    diff  += (q->dataf[i] - d) * (q->dataf[i] - d);
  }
  return diff / q->len;
}

static void best_quantize(quantized_t q, size_t from, size_t to, f32 target) {
  for (size_t nbit = from; nbit <= to; nbit++) {
    q->nbit = nbit;
    quantize(q);
    if (quantize_diff(q) < target) break;
  }
}

// ----------------------------------------------------------------------------------------------------
//; bits

#if defined(__GNUC__) && !defined(__clang__)
#  define clz(x)                                                                                   \
    _Generic((x),                                                                                  \
        unsigned char: __builtin_clz((uint)(x)) - 24,                                              \
        unsigned short: __builtin_clz((uint)(x)) - 16,                                             \
        unsigned int: __builtin_clz(x),                                                            \
        unsigned long: __builtin_clzl(x),                                                          \
        unsigned long long: __builtin_clzll(x))
#else
#  define clz(x)                                                                                   \
    _Generic((x),                                                                                  \
        unsigned char: __builtin_clzs((ushort)(x)) - 8,                                            \
        unsigned short: __builtin_clzs(x),                                                         \
        unsigned int: __builtin_clz(x),                                                            \
        unsigned long: __builtin_clzl(x),                                                          \
        unsigned long long: __builtin_clzll(x))
#endif

static u8 bit_reverse_8(u8 x) {
  x = ((x & 0x55) << 1) | ((x >> 1) & 0x55);
  x = ((x & 0x33) << 2) | ((x >> 2) & 0x33);
  x = ((x & 0x0f) << 4) | ((x >> 4) & 0x0f);
  return x;
}
static u16 bit_reverse_16(u16 x) {
  x = ((x & 0x5555) << 1) | ((x >> 1) & 0x5555);
  x = ((x & 0x3333) << 2) | ((x >> 2) & 0x3333);
  x = ((x & 0x0f0f) << 4) | ((x >> 4) & 0x0f0f);
  x = ((x & 0x00ff) << 8) | ((x >> 8) & 0x00ff);
  return x;
}
static u32 bit_reverse_32(u32 x) {
  x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555);
  x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333);
  x = ((x & 0x0f0f0f0f) << 4) | ((x >> 4) & 0x0f0f0f0f);
  x = ((x & 0x00ff00ff) << 8) | ((x >> 8) & 0x00ff00ff);
  x = (x << 16) | (x >> 16);
  return x;
}
static u64 bit_reverse_64(u64 x) {
  x = ((x & 0x5555555555555555) << 1) | ((x >> 1) & 0x5555555555555555);
  x = ((x & 0x3333333333333333) << 2) | ((x >> 2) & 0x3333333333333333);
  x = ((x & 0x0f0f0f0f0f0f0f0f) << 4) | ((x >> 4) & 0x0f0f0f0f0f0f0f0f);
  x = ((x & 0x00ff00ff00ff00ff) << 8) | ((x >> 8) & 0x00ff00ff00ff00ff);
  x = ((x & 0x0000ffff0000ffff) << 16) | ((x >> 16) & 0x0000ffff0000ffff);
  x = (x << 32) | (x >> 32);
  return x;
}
#define bit_reverse(x)                                                                             \
  _Generic((x),                                                                                    \
      u8: bit_reverse_8(x),                                                                        \
      u16: bit_reverse_16(x),                                                                      \
      u32: bit_reverse_32(x),                                                                      \
      u64: bit_reverse_64(x),                                                                      \
      i8: bit_reverse_8(x),                                                                        \
      i16: bit_reverse_16(x),                                                                      \
      i32: bit_reverse_32(x),                                                                      \
      i64: bit_reverse_64(x))

// ----------------------------------------------------------------------------------------------------
//; fft

#define I  1.i
#define PI 3.14159265358979323846264338327950288

#define FFT(_x_) fftf##_x_
typedef f32  FT;
typedef cf32 CT;

#define bit_rev(n) (bit_reverse((u64)(n)) >> (64 - log_n))

static CT   fft_wn[64] = {};
static CT   aft_wn[64] = {};
static bool fft_inited = false;

static void FFT(_init)() {
  if (fft_inited) return;
  fft_inited = true;
  f64 n      = 1;
  for (int i = 0; i < 64; i++) {
    cf64 x     = 6.283185307179586232i * n;
    fft_wn[i]  = __builtin_cexp(x);
    aft_wn[i]  = __builtin_cexp(-x);
    n         *= .5;
  }
}

static void FFT()(CT *x, CT *s, size_t l, bool r) {
  if (!fft_inited) FFT(_init)();

  int    log_n = sizeof(size_t) * 8 - clz(l - 1);
  size_t n     = 1 << log_n;

  for (int i = 0; i < l; i++)
    x[bit_rev(i)] = s[i];
  for (int i = l; i < n; i++)
    x[bit_rev(i)] = 0;

  CT *_wn = (r ? aft_wn : fft_wn);
  CT  w, wn;

  for (int s = 1; s <= log_n; s++) {
    int m = 1 << s;
    wn    = _wn[s];
    for (int k = 0; k < n; k += m) {
      w = 1;
      for (int j = 0; j < m / 2; j++) {
        CT t              = w * x[k + j + m / 2];
        CT u              = x[k + j];
        x[k + j]          = u + t;
        x[k + j + m / 2]  = u - t;
        w                *= wn;
      }
    }
  }

  if (!r) {
    FT d = 1. / n;
    for (int i = 0; i < l; i++)
      x[i] *= d;
  }
}

static CT *FFT(_a)(CT *s, size_t l, bool r) {
  int    log_n = sizeof(size_t) * 8 - clz(l - 1);
  size_t n     = 1 << log_n;

  CT *x = malloc(n * sizeof(CT));

  FFT()(x, s, l, r);

  return x;
}

#undef bit_rev

// ----------------------------------------------------------------------------------------------------
//; mdct

typedef struct mdctf {
  bool   inverse;
  size_t len;
  f32   *buf;
  f32   *block;
  size_t bufp;
  f32   *output;
  void (*callback)(f32 *block, void *userdata);
  void *userdata;
} *mdctf_t;

#define FFT(_x_)  fftf##_x_
#define MDCT(_x_) mdctf##_x_
typedef f32  FT;
typedef cf32 CT;

static FT *MDCT(_a)(FT *s, size_t N, bool r) {
  const FT freq   = 2 * PI / N;
  const FT cfreq  = __builtin_cos(freq);
  const FT sfreq  = __builtin_sin(freq);
  const FT cfreq8 = __builtin_cos(freq / 8);
  const FT sfreq8 = __builtin_sin(freq / 8);

  if (!r) {
    FT *x = malloc(N * sizeof(FT));
    for (size_t k = 0; k < N / 2; k++) {
      FT t         = PI * (k + 0.5) / N;
      FT S         = __builtin_sin(t);
      x[k]         = s[k] * S;
      x[N - 1 - k] = s[N - 1 - k] * S;
    }
    s = x;
  }

  CT *x = malloc(N / 4 * sizeof(CT));
  if (!r) {
    FT C = cfreq8, S = sfreq8;
    FT tempr, tempi;
    for (size_t i = 0; i < N / 4; i++) {
      size_t n = N / 2 - 1 - 2 * i;
      if (i < (N >> 3))
        tempr = s[N / 4 + n] + s[N + N / 4 - 1 - n];
      else
        tempr = s[N / 4 + n] - s[N / 4 - 1 - n];

      n = 2 * i;
      if (i < (N >> 3))
        tempi = s[N / 4 + n] - s[N / 4 - 1 - n];
      else
        tempi = s[N / 4 + n] + s[N + N / 4 - 1 - n];

      x[i] = (tempr * C + tempi * S) + (tempi * C - tempr * S) * I;

      const FT _C = C * cfreq - S * sfreq;
      const FT _S = S * cfreq + C * sfreq;
      C = _C, S = _S;
    }
  } else {
    FT C = cfreq8, S = sfreq8;
    for (size_t i = 0; i < N / 4; i++) {
      FT tempr = -s[2 * i];
      FT tempi = s[N / 2 - 1 - 2 * i];

      x[i] = (tempr * C - tempi * S) + (tempi * C + tempr * S) * I;

      const FT _C = C * cfreq - S * sfreq, _S = S * cfreq + C * sfreq;
      C = _C, S = _S;
    }
  }

  if (!r) free(s);

  CT *y = malloc(N / 4 * sizeof(CT));
  FFT()(y, x, N / 4, r);
  free(x);

  FT *z = malloc(N * sizeof(FT));
  if (!r) {
    FT C = cfreq8, S = sfreq8;
    for (size_t i = 0; i < N / 4; i++) {
      FT tempr = 2 * (__real__ y[i] * C + __imag__ y[i] * S);
      FT tempi = 2 * (__imag__ y[i] * C - __real__ y[i] * S);

      z[2 * i]             = -tempr;
      z[N / 2 - 1 - 2 * i] = tempi;

      z[N / 2 + 2 * i] = -tempi;
      z[N - 1 - 2 * i] = tempr;

      const FT _C = C * cfreq - S * sfreq;
      const FT _S = S * cfreq + C * sfreq;
      C = _C, S = _S;
    }
  } else {
    FT C = cfreq8, S = sfreq8;
    for (size_t i = 0; i < N / 4; i++) {
      FT tempr = 0.5 * (__real__ y[i] * C - __imag__ y[i] * S);
      FT tempi = 0.5 * (__imag__ y[i] * C + __real__ y[i] * S);

      z[N / 2 + N / 4 - 1 - 2 * i] = tempr;
      if (i < (N >> 3))
        z[N / 2 + N / 4 + 2 * i] = tempr;
      else
        z[2 * i - N / 4] = -tempr;

      z[N / 4 + 2 * i] = tempi;
      if (i < (N >> 3))
        z[N / 4 - 1 - 2 * i] = -tempi;
      else
        z[N / 4 + N - 1 - 2 * i] = tempi;

      const FT _C = C * cfreq - S * sfreq;
      const FT _S = S * cfreq + C * sfreq;
      C = _C, S = _S;
    }
  }
  free(y);

  if (r) {
    FT *x = malloc(N * sizeof(FT));
    for (size_t k = 0; k < N / 2; k++) {
      FT t         = PI * (k + 0.5) / N;
      FT S         = __builtin_sin(t);
      x[k]         = z[k] * S;
      x[N - 1 - k] = z[N - 1 - k] * S;
    }
    free(z);
    z = x;
  }

  return z;
}

static MDCT(_t) MDCT(_alloc)(size_t length, bool inverse, void (*callback)(FT *, void *)) {
  MDCT(_t) mdct = malloc(sizeof(*mdct));
  if (mdct == NULL) return NULL;
  mdct->inverse  = inverse;
  mdct->len      = length;
  mdct->buf      = malloc(length * sizeof(FT));
  mdct->block    = inverse ? malloc(length / 2 * sizeof(FT)) : NULL;
  mdct->bufp     = 0;
  mdct->output   = NULL;
  mdct->callback = callback;
  mdct->userdata = NULL;
  if (!inverse) {
    for (size_t i = 0; i < mdct->len / 2; i++) {
      mdct->buf[i] = 0;
    }
    mdct->bufp = mdct->len / 2;
  }
  return mdct;
}

static void MDCT(_free)(MDCT(_t) mdct) {
  free(mdct->buf);
  free(mdct->block);
  free(mdct->output);
  free(mdct);
}

static void MDCT(_do_mdct)(MDCT(_t) mdct) {
  for (size_t i = mdct->bufp; i < mdct->len; i++) {
    mdct->buf[i] = 0;
  }

  free(mdct->output);
  mdct->output = MDCT(_a)(mdct->buf, mdct->len, false);
  mdct->bufp   = mdct->len / 2;

  for (size_t i = 0; i < mdct->len / 2; i++) {
    mdct->buf[i] = mdct->buf[i + mdct->len / 2];
  }

  if (mdct->callback) mdct->callback(mdct->output, mdct->userdata);
}

static void MDCT(_do_imdct)(MDCT(_t) mdct) {
  for (size_t i = mdct->bufp; i < mdct->len / 2; i++) {
    mdct->buf[i] = 0;
  }

  bool first = mdct->output == NULL;
  if (!first) {
    for (size_t i = 0; i < mdct->len / 2; i++) {
      mdct->block[i] = mdct->output[i + mdct->len / 2];
    }
  }

  free(mdct->output);
  mdct->output = MDCT(_a)(mdct->buf, mdct->len, true);
  mdct->bufp   = 0;

  if (first) return;

  for (size_t i = 0; i < mdct->len / 2; i++) {
    mdct->block[i] += mdct->output[i];
  }

  if (mdct->callback) mdct->callback(mdct->block, mdct->userdata);
}

static void MDCT(_do)(MDCT(_t) mdct) {
  if (!mdct->inverse) {
    MDCT(_do_mdct)(mdct);
  } else {
    MDCT(_do_imdct)(mdct);
  }
}

static void MDCT(_put)(MDCT(_t) mdct, FT *data, size_t length) {
  for (size_t i = 0; i < length; i++) {
    mdct->buf[mdct->bufp++] = data[i];
    if (mdct->bufp == (mdct->inverse ? mdct->len / 2 : mdct->len)) MDCT(_do)(mdct);
  }
}

static void MDCT(_final)(MDCT(_t) mdct) {
  if (mdct->bufp > 0) MDCT(_do)(mdct);
  if (!mdct->inverse) MDCT(_do)(mdct);
}

static FT *MDCT(_block)(MDCT(_t) mdct) {
  return mdct->block;
}

// ----------------------------------------------------------------------------------------------------
//; 压缩

typedef struct plac_compress {
  mdctf_t          mdct;
  struct quantized q;
  mostream_t       stream;
} *plac_compress_t;

#define block_len 2048

static void _plac_compress_block(f32 *block, void *_plac);

static plac_compress_t plac_compress_alloc() {
  plac_compress_t plac = malloc(sizeof(struct plac_compress));
  if (plac == NULL) return NULL;
  plac->mdct           = mdctf_alloc(2 * block_len, false, _plac_compress_block);
  plac->mdct->userdata = plac;
  plac->q.max          = 0;
  plac->q.min          = 0;
  plac->q.mid          = 0;
  plac->q.nbit         = 0;
  plac->q.len          = block_len;
  plac->q.dataf        = NULL;
  plac->q.datai        = malloc(block_len * 2);
  plac->stream         = mostream_alloc(1024);
  if (plac->q.datai == NULL) {
    free(plac);
    return NULL;
  }
  return plac;
}

static void plac_compress_free(plac_compress_t plac) {
  if (plac == NULL) return;
  mdctf_free(plac->mdct);
  mostream_free(plac->stream);
  free(plac->q.datai);
  free(plac);
}

static void plac_write_header(plac_compress_t plac, u32 samplerate, u64 nsamples) {
  u32 magic = MAGIC32('p', 'l', 'a', 'c');
  mostream_write(plac->stream, &magic, 4);
  u16 version = 1;
  mostream_write(plac->stream, &version, 2);
  mostream_write(plac->stream, &samplerate, 4);
  mostream_write(plac->stream, &nsamples, 8);
}

static void plac_write_data(plac_compress_t plac, quantized_t q) {
  mostream_write(plac->stream, &q->nbit, 2);
  mostream_write(plac->stream, &q->max, 2);
  if (q->nbit == 0) return;
  mobitstream_t s = mobitstream_alloc(1024);
  mobitstream_write_bits(s, q->min, 16);
  mobitstream_write_bits(s, q->mid, 16);
  for (size_t i = 0; i < q->len; i++) {
    const i16 data = q->datai[i] - q->mid;
    mobitstream_write_bit(s, data != 0);
  }
  for (size_t i = 0; i < q->len; i++) {
    const i16 data = q->datai[i] - q->mid;
    if (data != 0) mobitstream_write_bits(s, data, q->nbit);
  }
  size_t size = s->size + (s->bit_pos ? 1 : 0);
  mostream_write(plac->stream, &size, 2);
  mostream_write(plac->stream, s->buf, size);
  mobitstream_free(s);
}

static void _plac_compress_block(f32 *block, void *_plac) {
  plac_compress_t plac = _plac;
  volume_fine_tuning(block, block_len);
  mulaw_compress(block, block_len);
  plac->q.dataf = block;
  best_quantize(&plac->q, 0, 15, .001953125);
  plac_write_data(plac, &plac->q);
}

static void plac_compress_block(plac_compress_t plac, f32 *block, size_t len) {
  mdctf_put(plac->mdct, block, len);
}

static void plac_compress_final(plac_compress_t plac) {
  mdctf_final(plac->mdct);
}

// ----------------------------------------------------------------------------------------------------
//; 编码

static char *replace_extension(const char *path, const char *new_extension) {
  char *new_path = malloc(strlen(path) + strlen(new_extension) + 1);
  if (!new_path) return NULL;

  char *last_dot = strrchr(path, '.');
  if (last_dot) {
    size_t base_length = last_dot - path;
    strncpy(new_path, path, base_length);
    new_path[base_length] = '\0';
  } else {
    strcpy(new_path, path);
  }

  strcat(new_path, new_extension);
  return new_path;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Usage: %s <input.mp3> [output.plac]\n", argv[0]);
    return 1;
  }

  plac_compress_t cctx = plac_compress_alloc();

  AVFormatContext *formatContext = NULL;
  if (avformat_open_input(&formatContext, argv[1], NULL, NULL) < 0) return 1;
  if (avformat_find_stream_info(formatContext, NULL) < 0) {
    avformat_close_input(&formatContext);
    return 1;
  }
  const AVCodec *codec = NULL;
  int            sid   = av_find_best_stream(formatContext, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
  if (sid < 0) {
    avformat_close_input(&formatContext);
    return 1;
  }
  AVCodecContext *codecContext = avcodec_alloc_context3(codec);
  if (codecContext == NULL) {
    avformat_close_input(&formatContext);
    return 1;
  }
  if (avcodec_parameters_to_context(codecContext, formatContext->streams[sid]->codecpar) < 0) {
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    return 1;
  }
  if (avcodec_open2(codecContext, codec, NULL) < 0) {
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    return 1;
  }
  if (codecContext->ch_layout.nb_channels > 2) {
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    return 1;
  }
  AVFrame *frame = av_frame_alloc();
  if (frame == NULL) {
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    return 1;
  }
  AVChannelLayout mono_layout;
  av_channel_layout_default(&mono_layout, 1);
  SwrContext *swr_ctx = NULL;
  swr_alloc_set_opts2(&swr_ctx, &mono_layout, AV_SAMPLE_FMT_FLT, codecContext->sample_rate,
                      &codecContext->ch_layout, codecContext->sample_fmt, codecContext->sample_rate,
                      0, NULL);
  if (!swr_ctx) {
    av_frame_free(&frame);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    return 1;
  }
  if (swr_init(swr_ctx) < 0) {
    swr_free(&swr_ctx);
    av_frame_free(&frame);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    return 1;
  }
  size_t bufsize = 4096;
  void  *buf     = malloc(bufsize);
  plac_write_header(cctx, codecContext->sample_rate, 0);
  for (AVPacket packet; av_read_frame(formatContext, &packet) >= 0; av_packet_unref(&packet)) {
    if (packet.stream_index != sid) continue;
    if (avcodec_send_packet(codecContext, &packet) < 0) break;
    while (avcodec_receive_frame(codecContext, frame) >= 0) {
      if (frame->nb_samples * 4 > bufsize) {
        bufsize = frame->nb_samples * 4;
        free(buf);
        buf = malloc(bufsize);
      }
      int out_samples = swr_convert(swr_ctx, (uint8_t *const *)&buf, frame->nb_samples,
                                    (const uint8_t **)frame->data, frame->nb_samples);
      if (out_samples <= 0) continue;
      plac_compress_block(cctx, buf, out_samples);
      av_frame_unref(frame);
    }
  }
  plac_compress_final(cctx);
  free(buf);
  swr_free(&swr_ctx);
  av_frame_free(&frame);
  avcodec_free_context(&codecContext);
  avformat_close_input(&formatContext);

  const char *path = argc == 3 ? argv[2] : replace_extension(argv[1], ".plac");

  write_to_file(path, cctx->stream->buf, cctx->stream->size);

  return 0;
}
