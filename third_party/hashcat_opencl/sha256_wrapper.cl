/**
 * sha256_wrapper.cl
 *
 * 使用 hashcat 原生的 SHA256 实现（inc_hash_sha256.*）的封装 kernel。
 *
 * 每个 work-item 处理一条消息：
 *
 *   参数:
 *     msgs        : 所有消息拼在一起的缓冲区，按字节存储（raw bytes）
 *                   在 host 侧保证每条消息所在区域零填充到固定 stride。
 *     msg_lens    : 每条消息的真实长度（字节数）
 *     msg_stride  : 每条消息在 msgs 中占用的跨度（单位：u32，也就是 4 字节一个单位）
 *     digests     : 输出，每条消息 8 个 u32（标准 SHA256 256-bit）
 *
 *   注意:
 *     - 长度 msg_lens[] 是“字节数”，跟 SHA-256 标准一致。
 *     - msg_stride 是“以 u32 为单位的跨度”，即 msg_stride_words。
 *       如果你 host 侧用的是字节数 stride_bytes，则有:
 *           msg_stride = stride_bytes / 4;
 */

#define IS_OPENCL 1  // 给 inc_vendor.h 一个环境标记（可选）

// ---- 在 OpenCL 里补上 stdint 风格类型，让 inc_types.h 不再报 uint8_t 未定义 ----
typedef uchar  uint8_t;
typedef ushort uint16_t;
typedef uint   uint32_t;
typedef ulong  uint64_t;

// ---- 引入 hashcat 的通用工具 & SHA256 实现 ----
// inc_common.cl 自己会 #include inc_vendor.h / inc_types.h / inc_platform.h / inc_common.h
#include "inc_common.cl"

// inc_hash_sha256.cl 会 #include inc_vendor.h / inc_types.h / inc_platform.h / inc_common.h / inc_hash_sha256.h
#include "inc_hash_sha256.cl"

// ---- 封装 kernel ----
// 使用 hashcat 自己定义的地址空间/修饰符: GLOBAL_AS / PRIVATE_AS / KERNEL_FQ / u32 / u8 等
KERNEL_FQ void sha256_wrapper (
  GLOBAL_AS const u32 *msgs,       // 注意：底层其实是 byte buffer，只是按 u32* 访问
  GLOBAL_AS const u32 *msg_lens,   // 每条消息长度（字节）
  const        u32    msg_stride,  // 每条消息占用的 u32 数（即 stride_bytes / 4）
  GLOBAL_AS       u32 *digests     // 输出：N * 8 个 u32
)
{
  const u32 gid = get_global_id (0);

  // 取出本条消息长度（字节）
  const u32 len = msg_lens[gid];

  // 计算本条消息对应的 u32* 起始位置
  GLOBAL_AS const u32 *w = msgs + ((size_t) gid * (size_t) msg_stride);

  // hashcat 的 SHA256 上下文
  sha256_ctx_t ctx;

  sha256_init (&ctx);

  // 关键点：用 hashcat 提供的 "global + swap" 版本，直接从 GLOBAL_AS 读取并做字节序转换
  // len 是字节数，w 是 4 字节对齐的 global 缓冲区
  sha256_update_global_swap (&ctx, w, (int) len);

  // 做最终的 padding + 长度写入 + transform
  sha256_final (&ctx);

  // 写回 8 × u32 的 digest
  GLOBAL_AS u32 *out = digests + ((size_t) gid * 8u);

  out[0] = ctx.h[0];
  out[1] = ctx.h[1];
  out[2] = ctx.h[2];
  out[3] = ctx.h[3];
  out[4] = ctx.h[4];
  out[5] = ctx.h[5];
  out[6] = ctx.h[6];
  out[7] = ctx.h[7];
}
