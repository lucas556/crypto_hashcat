// sha512_wrapper.cl
//
// 每个 work-item 对一条消息做 SHA512：
//   输入：
//     msgs      : GLOBAL u32*，所有消息按 stride 存放（字节对齐到 64）
//     lens      : GLOBAL u32*，每条消息的字节长度
//     msg_stride: 每条消息占用的 u32 数（= stride_bytes / 4）
//   输出：
//     digests   : GLOBAL u32*，每条 16 个 u32 = 64 字节 SHA512 摘要

#include "inc_platform.cl"
#include "inc_vendor.h"
#include "inc_types.h"
#include "inc_common.cl"
#include "inc_hash_sha512.cl"

__kernel void sha512_wrapper (__global const u32 *msgs,
                              __global const u32 *lens,
                              const u32          msg_stride,
                              __global       u32 *digests)
{
  const u32 gid = get_global_id (0);

  const u32 len_bytes = lens[gid];          // 消息字节数
  const u32 offs      = gid * msg_stride;   // 该消息在 msgs 中的 u32 起始下标

  sha512_ctx_t ctx;

  sha512_init (&ctx);

  // 和 sha256_wrapper 一样用 *_global_swap 版本，把 ASCII/小端字节按大端喂给 sha512
  if (len_bytes > 0)
  {
    sha512_update_global_swap (&ctx, msgs + offs, (int) len_bytes);
  }

  sha512_final (&ctx);

  // ctx.h 是 8 x u64，总计 64 字节；我们拆成 16 个 u32 存入 digests：
  //   out[0]..out[15]：按 (h0_hi, h0_lo, h1_hi, h1_lo, ..., h7_hi, h7_lo)
  u32 *out = digests + gid * 16u;

  for (int i = 0; i < 8; i++)
  {
    const u64 v = ctx.h[i];

    out[2 * i + 0] = (u32) (v >> 32);         // 高 32 位
    out[2 * i + 1] = (u32) (v & 0xffffffffu); // 低 32 位
  }
}
