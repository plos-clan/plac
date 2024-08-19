#include <alsa/asoundlib.h>
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
//; 文件读取

static void *read_from_file(const char *filename, size_t *size) {
  if (filename == NULL) return NULL;

  int fd = open(filename, O_RDONLY);
  if (fd < 0) return NULL;

  struct stat file_stat;
  if (fstat(fd, &file_stat) < 0) {
    close(fd);
    return NULL;
  }
  if (size) *size = file_stat.st_size;

  void *data = malloc(file_stat.st_size);
  if (data == NULL) return NULL;

  int rets = read(fd, data, file_stat.st_size);
  close(fd);
  if (rets != file_stat.st_size) {
    free(data);
    return NULL;
  }

  return data;
}

// ----------------------------------------------------------------------------------------------------
//; 内存流

typedef struct mistream {
  const void *buf;
  size_t      size;
  size_t      pos;
} *mistream_t;

static mistream_t mistream_alloc(const void *buffer, size_t size) {
  mistream_t stream = malloc(sizeof(struct mistream));
  if (stream == NULL) return NULL;
  stream->buf  = buffer;
  stream->size = size;
  stream->pos  = 0;
  return stream;
}

static void mistream_free(mistream_t stream) {
  if (stream == NULL) return;
  free(stream);
}

static size_t mistream_read(mistream_t stream, void *data, size_t size) {
  if (stream->pos + size > stream->size) size = stream->size - stream->pos;
  memcpy(data, stream->buf + stream->pos, size);
  stream->pos += size;
  return size;
}

typedef struct mibitstream {
  const void *buf;
  size_t      size;
  size_t      pos;
  size_t      bit_pos;
} *mibitstream_t;

static mibitstream_t mibitstream_alloc(const void *buffer, size_t size) {
  mibitstream_t stream = malloc(sizeof(struct mibitstream));
  if (stream == NULL) return NULL;
  stream->buf     = buffer;
  stream->size    = size;
  stream->pos     = 0;
  stream->bit_pos = 0;
  return stream;
}

static void mibitstream_free(mibitstream_t stream) {
  if (stream == NULL) return;
  free(stream);
}

static int mibitstream_read_bit(mibitstream_t stream) {
  if (stream->pos >= stream->size) return -1;
  uint8_t byte = ((uint8_t *)stream->buf)[stream->pos];
  int     bit  = (byte >> (7 - stream->bit_pos)) & 1;
  stream->bit_pos++;
  if (stream->bit_pos == 8) {
    stream->bit_pos = 0;
    stream->pos++;
  }
  return bit;
}

static size_t mibitstream_read_bits(mibitstream_t stream, size_t nbits) {
  size_t bits = 0;
  for (size_t i = 0; i < nbits; i++) {
    int bit = mibitstream_read_bit(stream);
    if (bit == -1) return 0;
    bits |= (bit << i);
  }
  return bits;
}

static ssize_t mibitstream_read_bitsi(mibitstream_t stream, size_t nbits) {
  ssize_t bits = 0;
  for (size_t i = 0; i < nbits - 1; i++) {
    int bit = mibitstream_read_bit(stream);
    if (bit == -1) return 0;
    bits |= (bit << i);
  }
  int sign = mibitstream_read_bit(stream);
  if (sign == -1) return 0;
  return sign ? ~(((size_t)1 << (nbits - 1)) - 1) | bits : bits;
}

// ----------------------------------------------------------------------------------------------------
//; mulaw

#define MU 1023

static void mulaw_expand(f32 *data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    f32  x    = data[i];
    bool sign = x < 0;
    if (sign) x = -x;
    x       = (__builtin_pow(1 + MU, x) - 1) / MU;
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

void dequantize(quantized_t q) {
  f32 max = (f32)q->max / 256;
  f32 min = (f32)q->min / 256;
  if (q->nbit == 0) {
    for (size_t i = 0; i < q->len; i++) {
      q->dataf[i] = min;
    }
    return;
  }
  f32 k = (max - min) / ((1 << q->nbit) - 1);
  for (size_t i = 0; i < q->len; i++) {
    q->dataf[i] = q->datai[i] * k + min;
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
//; 解压缩

typedef void (*cb_plac_decompress_t)(f32 *block, size_t len, void *userdata);

typedef struct plac_decompress {
  size_t               block_len;
  mdctf_t              mdct;
  struct quantized     q;
  mistream_t           stream;
  cb_plac_decompress_t callback;
  void                *userdata;
} *plac_decompress_t;

void _plac_decompress_block(f32 *block, void *_plac);

plac_decompress_t plac_decompress_alloc(const void *buffer, size_t size, size_t block_len) {
  plac_decompress_t plac = malloc(sizeof(struct plac_decompress));
  if (plac == NULL) return NULL;
  plac->block_len      = block_len;
  plac->mdct           = mdctf_alloc(2 * block_len, true, _plac_decompress_block);
  plac->mdct->userdata = plac;
  plac->q.max          = 0;
  plac->q.min          = 0;
  plac->q.mid          = 0;
  plac->q.nbit         = 0;
  plac->q.len          = block_len;
  plac->q.dataf        = malloc(block_len * 4);
  plac->q.datai        = malloc(block_len * 2);
  plac->stream         = mistream_alloc(buffer, size);
  if (plac->q.datai == NULL) {
    free(plac);
    return NULL;
  }
  return plac;
}

void plac_decompress_free(plac_decompress_t plac) {
  if (plac == NULL) return;
  mdctf_free(plac->mdct);
  mistream_free(plac->stream);
  free(plac->q.dataf);
  free(plac->q.datai);
  free(plac);
}

bool plac_read_header(plac_decompress_t plac, u16 *samplerate, u32 *nsamples) {
  u32 magic;
  mistream_read(plac->stream, &magic, 4);
  if (magic != MAGIC32('p', 'l', 'a', 'c')) return false;
  u16 version;
  mistream_read(plac->stream, &version, 2);
  if (version != 0) return false;
  mistream_read(plac->stream, samplerate, 2);
  mistream_read(plac->stream, nsamples, 4);
  return true;
}

void plac_read_data(plac_decompress_t plac, quantized_t q) {
  mistream_read(plac->stream, &q->nbit, 2);
  mistream_read(plac->stream, &q->max, 2);
  if (q->nbit == 0) return;
  size_t size = 0;
  mistream_read(plac->stream, &size, 2);
  void *buf = malloc(size);
  mistream_read(plac->stream, buf, size);
  mibitstream_t s = mibitstream_alloc(buf, size);
  q->min          = mibitstream_read_bits(s, 16);
  q->mid          = mibitstream_read_bits(s, 16);
  bool *bitmap    = malloc(q->len);
  for (size_t i = 0; i < q->len; i++) {
    bitmap[i] = mibitstream_read_bit(s);
  }
  for (size_t i = 0; i < q->len; i++) {
    const i16 data = bitmap[i] ? mibitstream_read_bitsi(s, q->nbit) : 0;
    q->datai[i]    = data + q->mid;
  }
  free(bitmap);
  mibitstream_free(s);
  free(buf);
}

void _plac_decompress_block(f32 *block, void *_plac) {
  plac_decompress_t plac = _plac;
  if (plac->callback) plac->callback(block, plac->block_len, plac->userdata);
}

bool plac_decompress_block(plac_decompress_t plac) {
  if (plac->stream->pos == plac->stream->size) return false;
  plac_read_data(plac, &plac->q);
  dequantize(&plac->q);
  mulaw_expand(plac->q.dataf, plac->block_len);
  mdctf_put(plac->mdct, plac->q.dataf, plac->block_len);
  return true;
}

// ----------------------------------------------------------------------------------------------------
//; 播放

snd_pcm_t *pcm_out;

#define N 1024 // 必须为 1024

void play_audio(f32 *block, size_t len, void *userdata) {
  f32 volume = *(f32 *)userdata;
  if (volume != 1) {
    for (size_t i = 0; i < len; i++) {
      block[i] *= volume;
    }
  }
  snd_pcm_writei(pcm_out, block, len);
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Usage: %s <input.plac> [volume]\n", argv[0]);
    return 1;
  }

  size_t bufsize;
  void  *buf = read_from_file(argv[1], &bufsize);
  if (buf == NULL) {
    fprintf(stderr, "Failed to read file\n");
    return 1;
  }

  f32 volume = 1;
  if (argc == 3) {
    volume = atof(argv[2]);
    if (volume < 0 || volume > 2) {
      fprintf(stderr, "Volume must be between 0 and 2\n");
      return 1;
    }
  }

  plac_decompress_t dctx = plac_decompress_alloc(buf, bufsize, N);
  dctx->callback         = play_audio;
  dctx->userdata         = &volume;

  u16 samplerate;
  u32 nsamples;
  plac_read_header(dctx, &samplerate, &nsamples);

  snd_pcm_open(&pcm_out, "default", SND_PCM_STREAM_PLAYBACK, 0);
  snd_pcm_set_params(pcm_out, SND_PCM_FORMAT_FLOAT, SND_PCM_ACCESS_RW_INTERLEAVED, 1, samplerate, 0,
                     .5e6);

  while (plac_decompress_block(dctx)) {}

  snd_pcm_close(pcm_out);

  return 0;
}
