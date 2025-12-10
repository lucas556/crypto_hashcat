/* 
 * Minimal SHA-256 wrapper kernel using hashcat's inc_hash_sha256
 * 由 Lucas 项目封装，仅调用 hashcat 的实现，不改动其内部算法
 */

#ifndef SHA256_WRAPPER_CL
#define SHA256_WRAPPER_CL

// === hashcat 基础依赖（原样保留，不修改这些文件的内容） ===
#include "inc_vendor.h"
#include "inc_types.h"
#include "inc_platform.cl"
#include "inc_common.cl"
#include "inc_scalar.cl"

// === SHA256 实现（hashcat 原始文件） ===
#include "inc_hash_sha256.h"
#include "inc_hash_sha256.cl"

// ====== 最小封装：对每个 work-item 计算 1 个 SHA256(msg) ======
//
// 入参：
//   msgs        : 所有消息拼在一起的缓冲区（按 u32 存储）
//   msg_lens    : 每条消息的长度（单位：字节）
//   msg_stride  : 每条消息在 msgs 中占用的 u32 数量（即 stride，以 u32 为单位）
//   digests     : 输出，8 * u32/条消息（标准 SHA256 8 个 32bit 大端字）
//
// 注意：
//   1) msgs 按 hashcat 的约定存成 u32 数组：每 4 字节打包成一个 u32，
//      字节序需和你选用的 *_swap 辅助函数保持一致（通常是 big-endian）。
//   2) sha256_update(ctx, ptr, len) 的 len 参数是 “字节数”，不是 u32 个数。
//   3) 如果消息长度 > 64 字节，会循环多次调用 sha256_update。
//
// 主机侧你只需要：
//   - 把每条消息 pack 成 u32 数组（补零到 msg_stride*4 字节）
//   - 填好 msg_lens[i]（真实字节长度）
//   - 设好 msg_stride（单位是 u32）
//   - 全局 work-items 数量 >= 消息条数
//

__kernel void sha256_wrapper(__global const u32 *msgs,
                             __global const u32 *msg_lens,
                             const uint          msg_stride,   // stride in u32
                             __global       u32 *digests)      // 8 u32 per msg
{
  const uint gid = get_global_id(0);

  // 读取本条消息的长度（字节）
  const u32 len_bytes = msg_lens[gid];

  // 计算本条消息在 msgs 中的起始指针（按 u32 下标）
  const __global u32 *msg_base = msgs + ((size_t)gid * (size_t)msg_stride);

  // 初始化 ctx
  sha256_ctx_t ctx;
  sha256_init(&ctx);

  // 按 64 字节块迭代调用 sha256_update
  u32 remaining = len_bytes;
  u32 offset_bytes = 0;

  while (remaining > 64)
  {
    const u32 chunk_len = 64;        // 每次整块 64 字节
    const u32 word_off  = offset_bytes >> 2;  // byte → u32 下标

    sha256_update(&ctx, msg_base + word_off, chunk_len);

    offset_bytes += chunk_len;
    remaining    -= chunk_len;
  }

  // 处理最后不足 64 字节的尾块（如果为 0 就不调用）
  if (remaining > 0)
  {
    const u32 word_off = offset_bytes >> 2;
    sha256_update(&ctx, msg_base + word_off, remaining);
  }

  // 结束，ctx.h[0..7] 即 8×u32 的 digest
  sha256_final(&ctx);

  __global u32 *out = digests + ((size_t)gid * 8u);

  out[0] = ctx.h[0];
  out[1] = ctx.h[1];
  out[2] = ctx.h[2];
  out[3] = ctx.h[3];
  out[4] = ctx.h[4];
  out[5] = ctx.h[5];
  out[6] = ctx.h[6];
  out[7] = ctx.h[7];
}

#endif // SHA256_WRAPPER_CL
