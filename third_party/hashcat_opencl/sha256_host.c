// sha256_host.c  (batch 版本)
//
// 对输入文件每一行做 SHA256（不含 '\n'），输出到结果文件。
// 使用 hashcat 的 sha256_wrapper.cl + inc_* 实现。
// 支持分批处理超大文件，避免一次性分配几十 GB 内存。

#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

/*
 * 在很多系统里，如果 CL_TARGET_OPENCL_VERSION < 200，
 * 头文件不会声明 cl_queue_properties 和 clCreateCommandQueueWithProperties。
 * 这里自己补上 typedef + 原型，方便在 OpenCL 1.2 头文件下使用 2.0 API。
 */
#ifndef CL_VERSION_2_0
typedef cl_bitfield cl_queue_properties;
extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueueWithProperties (cl_context                 context,
                                    cl_device_id               device,
                                    const cl_queue_properties *properties,
                                    cl_int                    *errcode_ret);
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHECK_CL(err, msg) \
  do { \
    if ((err) != CL_SUCCESS) { \
      fprintf(stderr, "%s failed with error %d\n", (msg), (err)); \
      exit(1); \
    } \
  } while (0)

// 一批最多处理多少行（可根据内存调大或调小）
#define MAX_BATCH_LINES 50000000u  // 5000 万

// 读取 CL 源码文件
static char *read_text_file (const char *path, size_t *out_size)
{
  FILE *f = fopen (path, "rb");
  if (!f)
  {
    perror (path);
    exit (1);
  }

  if (fseek (f, 0, SEEK_END) != 0)
  {
    perror ("fseek");
    exit (1);
  }

  long len = ftell (f);
  if (len < 0)
  {
    perror ("ftell");
    exit (1);
  }
  rewind (f);

  char *buf = (char *) malloc ((size_t) len + 1);
  if (!buf)
  {
    fprintf (stderr, "malloc failed for %s\n", path);
    exit (1);
  }

  size_t nread = fread (buf, 1, (size_t) len, f);
  fclose (f);

  buf[nread] = '\0';
  if (out_size) *out_size = nread;
  return buf;
}

// 打印平台和设备信息
static void print_platform_device_info (cl_platform_id platform, cl_device_id device)
{
  char buf[256];
  size_t sz = 0;

  if (clGetPlatformInfo (platform, CL_PLATFORM_NAME, sizeof (buf), buf, &sz) == CL_SUCCESS)
  {
    if (sz >= sizeof (buf)) sz = sizeof (buf) - 1;
    buf[sz] = '\0';
    fprintf (stderr, "[OpenCL] Platform: %s\n", buf);
  }

  if (clGetDeviceInfo (device, CL_DEVICE_NAME, sizeof (buf), buf, &sz) == CL_SUCCESS)
  {
    if (sz >= sizeof (buf)) sz = sizeof (buf) - 1;
    buf[sz] = '\0';
    fprintf (stderr, "[OpenCL] Device  : %s\n", buf);
  }
}

// digest[32] -> hex[65]
static void digest_to_hex (const uint8_t *digest, char *hex_out)
{
  static const char hexdig[] = "0123456789abcdef";
  for (int i = 0; i < 32; i++)
  {
    unsigned v = digest[i];
    hex_out[i*2+0] = hexdig[v >> 4];
    hex_out[i*2+1] = hexdig[v & 0xF];
  }
  hex_out[64] = '\0';
}

int main (int argc, char **argv)
{
  if (argc != 3)
  {
    fprintf (stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    return 1;
  }

  const char *input_path  = argv[1];
  const char *output_path = argv[2];

  cl_int err;

  // 0. 打开输入 / 输出文件
  FILE *fin = fopen (input_path, "rb");
  if (!fin)
  {
    perror (input_path);
    return 1;
  }

  FILE *fout = fopen (output_path, "wb");
  if (!fout)
  {
    perror (output_path);
    fclose (fin);
    return 1;
  }

  // 1. 选择 OpenCL 平台 & 设备
  cl_uint num_platforms = 0;
  CHECK_CL (clGetPlatformIDs (0, NULL, &num_platforms), "clGetPlatformIDs(count)");

  if (num_platforms == 0)
  {
    fprintf (stderr, "No OpenCL platforms found\n");
    fclose (fin);
    fclose (fout);
    return 1;
  }

  cl_platform_id *platforms =
      (cl_platform_id *) malloc (sizeof (cl_platform_id) * num_platforms);
  CHECK_CL (clGetPlatformIDs (num_platforms, platforms, NULL),
            "clGetPlatformIDs(list)");

  cl_platform_id chosen_platform = NULL;
  cl_device_id   chosen_device   = NULL;

  for (cl_uint pi = 0; pi < num_platforms; pi++)
  {
    cl_platform_id p = platforms[pi];

    cl_uint num_devices = 0;
    cl_int err_local;

    // 优先 GPU，再 CPU，最后 ALL
    err_local = clGetDeviceIDs (p, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err_local == CL_DEVICE_NOT_FOUND || num_devices == 0)
    {
      err_local = clGetDeviceIDs (p, CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);
    }
    if (err_local == CL_DEVICE_NOT_FOUND || num_devices == 0)
    {
      err_local = clGetDeviceIDs (p, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    }

    if (err_local != CL_SUCCESS || num_devices == 0)
    {
      continue;
    }

    cl_device_id *devices =
        (cl_device_id *) malloc (sizeof (cl_device_id) * num_devices);
    CHECK_CL (clGetDeviceIDs (p, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL),
              "clGetDeviceIDs(list)");

    chosen_platform = p;
    chosen_device   = devices[0];

    free (devices);
    break;
  }

  if (chosen_platform == NULL || chosen_device == NULL)
  {
    fprintf (stderr, "No OpenCL devices found on any platform\n");
    free (platforms);
    fclose (fin);
    fclose (fout);
    return 1;
  }

  cl_platform_id platform = chosen_platform;
  cl_device_id   device   = chosen_device;

  free (platforms);

  print_platform_device_info (platform, device);

  // 2. 创建 context & queue（带 profiling）
  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_CL (err, "clCreateContext");

  const cl_queue_properties props[] =
  {
    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
    0
  };

  cl_command_queue queue =
      clCreateCommandQueueWithProperties (context, device, props, &err);
  CHECK_CL (err, "clCreateCommandQueueWithProperties");

  // 3. 读取 & 编译 sha256_wrapper.cl （只编译一次）
  size_t src_size = 0;
  char *src = read_text_file ("sha256_wrapper.cl", &src_size);

  const char  *sources[] = { src };
  const size_t lengths[] = { src_size };

  cl_program program =
      clCreateProgramWithSource (context, 1, sources, lengths, &err);
  CHECK_CL (err, "clCreateProgramWithSource");

  const char *build_opts = NULL; // 如需 -I/path/to/hashcat/OpenCL 在这里加

  err = clBuildProgram (program, 1, &device, build_opts, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t log_size = 0;
    clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                           0, NULL, &log_size);

    char *log = (char *) malloc (log_size + 1);
    clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                           log_size, log, NULL);
    log[log_size] = '\0';

    fprintf (stderr, "Build failed:\n%s\n", log);
    free (log);
    CHECK_CL (err, "clBuildProgram");
  }

  free (src);

  cl_kernel kernel = clCreateKernel (program, "sha256_wrapper", &err);
  CHECK_CL (err, "clCreateKernel");

  // 4. 为 batch 分配复用的 host 侧辅助数组
  uint32_t max_batch_lines = MAX_BATCH_LINES;

  char     **line_bufs = (char   **) malloc (max_batch_lines * sizeof (char *));
  uint32_t *lens_host  = (uint32_t *) malloc (max_batch_lines * sizeof (uint32_t));
  if (!line_bufs || !lens_host)
  {
    fprintf (stderr, "malloc failed for batch metadata\n");
    fclose (fin);
    fclose (fout);
    return 1;
  }

  // 用于 getline
  char  *line    = NULL;
  size_t linecap = 0;

  // 统计整体性能
  double              total_kernel_time_s = 0.0;
  unsigned long long  total_msgs          = 0ULL;
  unsigned int        batch_index         = 0;

  for (;;)
  {
    // 5. 读一批行
    uint32_t num_msgs = 0;
    size_t   max_len  = 0;

    for (;;)
    {
      if (num_msgs >= max_batch_lines) break;

      ssize_t linelen = getline (&line, &linecap, fin);
      if (linelen < 0) break; // EOF

      size_t len = (size_t) linelen;

      // 去掉末尾 '\n'
      if (len > 0 && line[len - 1] == '\n')
      {
        len--;
      }

      char *copy = (char *) malloc (len > 0 ? len : 1);
      if (!copy)
      {
        fprintf (stderr, "malloc failed for line copy\n");
        exit (1);
      }
      if (len > 0) memcpy (copy, line, len);

      line_bufs[num_msgs] = copy;
      lens_host[num_msgs] = (uint32_t) len;

      if (len > max_len) max_len = len;

      num_msgs++;
    }

    if (num_msgs == 0)
    {
      break; // 没有更多行
    }

    batch_index++;

    // 6. 为这一批计算 stride_bytes & 分配 msgs_bytes
    size_t stride_bytes;
    if (max_len == 0)
    {
      stride_bytes = 64;
    }
    else
    {
      // 仍然对齐到 64 bytes
      stride_bytes = ((max_len + 63) / 64) * 64;
    }

    uint32_t msg_stride = (uint32_t) (stride_bytes / 4);
    size_t   total_bytes  = (size_t) num_msgs * stride_bytes;

    fprintf (stderr,
             "[OpenCL] Batch %u: %u messages, max_len=%zu, stride_bytes=%zu (msg_stride=%u)\n",
             batch_index, num_msgs, max_len, stride_bytes, msg_stride);

    unsigned char *msgs_bytes = (unsigned char *) malloc (total_bytes);
    if (!msgs_bytes)
    {
      fprintf (stderr, "malloc failed for msgs_bytes (%zu bytes) in batch %u\n",
               total_bytes, batch_index);
      exit (1);
    }
    memset (msgs_bytes, 0, total_bytes);

    // 把每行复制到本批 msgs_bytes
    for (uint32_t k = 0; k < num_msgs; k++)
    {
      uint32_t len_k = lens_host[k];
      if (len_k > 0)
      {
        unsigned char *dst = msgs_bytes + (size_t) k * stride_bytes;
        unsigned char *src = (unsigned char *) line_bufs[k];
        memcpy (dst, src, len_k);
      }
    }

    // 7. 创建本批的 OpenCL buffers
    cl_mem buf_msgs = clCreateBuffer (
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total_bytes,
        msgs_bytes,
        &err);
    CHECK_CL (err, "clCreateBuffer(buf_msgs)");

    cl_mem buf_lens = clCreateBuffer (
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t) num_msgs * sizeof (uint32_t),
        lens_host,
        &err);
    CHECK_CL (err, "clCreateBuffer(buf_lens)");

    cl_mem buf_out = clCreateBuffer (
        context,
        CL_MEM_WRITE_ONLY,
        (size_t) num_msgs * 8u * sizeof (uint32_t),
        NULL,
        &err);
    CHECK_CL (err, "clCreateBuffer(buf_out)");

    free (msgs_bytes);

    // 8. 设置 kernel 参数（注意每批 msg_stride 不同，要重新 set）
    int arg = 0;
    CHECK_CL (clSetKernelArg (kernel, arg++, sizeof (cl_mem),   &buf_msgs),
              "clSetKernelArg(msgs)");
    CHECK_CL (clSetKernelArg (kernel, arg++, sizeof (cl_mem),   &buf_lens),
              "clSetKernelArg(lens)");
    CHECK_CL (clSetKernelArg (kernel, arg++, sizeof (uint32_t), &msg_stride),
              "clSetKernelArg(msg_stride)");
    CHECK_CL (clSetKernelArg (kernel, arg++, sizeof (cl_mem),   &buf_out),
              "clSetKernelArg(digests)");

    // 9. 启动 kernel + profiling
    size_t   global_work_size[1] = { (size_t) num_msgs };
    cl_event kernel_event;

    CHECK_CL (clEnqueueNDRangeKernel (queue, kernel, 1, NULL,
                                      global_work_size, NULL,
                                      0, NULL, &kernel_event),
              "clEnqueueNDRangeKernel");

    CHECK_CL (clWaitForEvents (1, &kernel_event), "clWaitForEvents");
    CHECK_CL (clFinish (queue), "clFinish");

    cl_ulong time_start = 0, time_end = 0;
    CHECK_CL (clGetEventProfilingInfo (kernel_event,
                                       CL_PROFILING_COMMAND_START,
                                       sizeof (time_start), &time_start, NULL),
              "clGetEventProfilingInfo(START)");
    CHECK_CL (clGetEventProfilingInfo (kernel_event,
                                       CL_PROFILING_COMMAND_END,
                                       sizeof (time_end), &time_end, NULL),
              "clGetEventProfilingInfo(END)");

    double kernel_time_ns = (double) (time_end - time_start);
    double kernel_time_s  = kernel_time_ns * 1e-9;

    total_kernel_time_s += kernel_time_s;
    total_msgs          += num_msgs;

    double hps  = (kernel_time_s > 0.0) ? ((double) num_msgs / kernel_time_s) : 0.0;
    double mhps = hps / 1e6;

    fprintf (stderr,
             "[OpenCL] Batch %u: kernel time = %.3f ms, speed = %.2f MH/s (%.3e H/s)\n",
             batch_index, kernel_time_s * 1e3, mhps, hps);

    clReleaseEvent (kernel_event);

    // 10. 读回 digest，写出到输出文件
    uint32_t *digests_host =
        (uint32_t *) malloc ((size_t) num_msgs * 8u * sizeof (uint32_t));
    if (!digests_host)
    {
      fprintf (stderr, "malloc failed for digests_host in batch %u\n", batch_index);
      exit (1);
    }

    CHECK_CL (clEnqueueReadBuffer (queue, buf_out, CL_TRUE, 0,
                                   (size_t) num_msgs * 8u * sizeof (uint32_t),
                                   digests_host,
                                   0, NULL, NULL),
              "clEnqueueReadBuffer");

    char hex[64 + 1];

    for (uint32_t k = 0; k < num_msgs; k++)
    {
      uint32_t *d = digests_host + (size_t) k * 8u;
      uint8_t   digest[32];

      // 以大端方式写入（和标准 SHA256 一致）
      for (int j = 0; j < 8; j++)
      {
        uint32_t v = d[j];
        digest[j*4 + 0] = (uint8_t)((v >> 24) & 0xff);
        digest[j*4 + 1] = (uint8_t)((v >> 16) & 0xff);
        digest[j*4 + 2] = (uint8_t)((v >>  8) & 0xff);
        digest[j*4 + 3] = (uint8_t)( v        & 0xff);
      }

      digest_to_hex (digest, hex);
      fprintf (fout, "%s\n", hex);
    }

    // 可选：打印本批第一条做 sanity check
    if (batch_index == 1 && num_msgs > 0)
    {
      uint32_t *d0 = digests_host + 0 * 8u;
      uint8_t   digest0[32];
      for (int j = 0; j < 8; j++)
      {
        uint32_t v = d0[j];
        digest0[j*4 + 0] = (uint8_t)((v >> 24) & 0xff);
        digest0[j*4 + 1] = (uint8_t)((v >> 16) & 0xff);
        digest0[j*4 + 2] = (uint8_t)((v >>  8) & 0xff);
        digest0[j*4 + 3] = (uint8_t)( v        & 0xff);
      }
      char hex0[65];
      digest_to_hex (digest0, hex0);
      fprintf (stderr, "[OpenCL] First line SHA256 = %s\n", hex0);
    }

    free (digests_host);

    clReleaseMemObject (buf_msgs);
    clReleaseMemObject (buf_lens);
    clReleaseMemObject (buf_out);

    // 释放这一批的行缓冲
    for (uint32_t k = 0; k < num_msgs; k++)
    {
      free (line_bufs[k]);
      line_bufs[k] = NULL;
    }
  }

  // 整体速度统计
  if (total_msgs > 0 && total_kernel_time_s > 0.0)
  {
    double hps  = (double) total_msgs / total_kernel_time_s;
    double mhps = hps / 1e6;

    fprintf (stderr,
             "[OpenCL] TOTAL: messages = %llu, kernel time = %.3f ms, speed = %.2f MH/s (%.3e H/s)\n",
             (unsigned long long) total_msgs,
             total_kernel_time_s * 1e3,
             mhps, hps);
  }

  free (line_bufs);
  free (lens_host);
  free (line);
  fclose (fin);
  fclose (fout);

  clReleaseKernel (kernel);
  clReleaseProgram (program);
  clReleaseCommandQueue (queue);
  clReleaseContext (context);

  return 0;
}
