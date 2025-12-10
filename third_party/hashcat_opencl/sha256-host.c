// host.c
// 用 OpenCL 调用 sha256_wrapper，计算输入文件的 SHA256
// 用法: ./host input_file output_file

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

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

// 读取文本/CL 源码文件到内存
static char *read_text_file(const char *path, size_t *out_size)
{
  FILE *f = fopen(path, "rb");
  if (!f)
  {
    perror(path);
    exit(1);
  }

  if (fseek(f, 0, SEEK_END) != 0)
  {
    perror("fseek");
    exit(1);
  }

  long len = ftell(f);
  if (len < 0)
  {
    perror("ftell");
    exit(1);
  }
  rewind(f);

  char *buf = (char *)malloc((size_t)len + 1);
  if (!buf)
  {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  size_t nread = fread(buf, 1, (size_t)len, f);
  fclose(f);

  buf[nread] = '\0';
  if (out_size) *out_size = nread;
  return buf;
}

// 读取任意二进制文件到内存
static unsigned char *read_binary_file(const char *path, size_t *out_size)
{
  FILE *f = fopen(path, "rb");
  if (!f)
  {
    perror(path);
    exit(1);
  }

  if (fseek(f, 0, SEEK_END) != 0)
  {
    perror("fseek");
    exit(1);
  }

  long len = ftell(f);
  if (len < 0)
  {
    perror("ftell");
    exit(1);
  }
  rewind(f);

  unsigned char *buf = (unsigned char *)malloc((size_t)len);
  if (!buf)
  {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  size_t nread = fread(buf, 1, (size_t)len, f);
  fclose(f);

  if (out_size) *out_size = nread;
  return buf;
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    return 1;
  }

  const char *input_path  = argv[1];
  const char *output_path = argv[2];

  cl_int err;

  // 1. 读取输入文件
  size_t file_size = 0;
  unsigned char *file_data = read_binary_file(input_path, &file_size);

  if (file_size == 0)
  {
    fprintf(stderr, "Input file is empty\n");
    free(file_data);
    return 1;
  }

  // 2. OpenCL 平台 & 设备
  cl_uint num_platforms = 0;
  CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs(count)");

  if (num_platforms == 0)
  {
    fprintf(stderr, "No OpenCL platforms found\n");
    free(file_data);
    return 1;
  }

  cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  CHECK_CL(clGetPlatformIDs(num_platforms, platforms, NULL), "clGetPlatformIDs(list)");

  cl_platform_id platform = platforms[0]; // 简单起见，取第一个
  free(platforms);

  cl_uint num_devices = 0;
  CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices), "clGetDeviceIDs(count)");

  if (num_devices == 0)
  {
    fprintf(stderr, "No OpenCL devices found\n");
    free(file_data);
    return 1;
  }

  cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL), "clGetDeviceIDs(list)");

  cl_device_id device = devices[0]; // 取第一个
  free(devices);

  // 3. 创建上下文 & 队列
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_CL(err, "clCreateContext");

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_CL(err, "clCreateCommandQueue");

  // 4. 读取 & 编译 program
  size_t src_size = 0;
  char *src = read_text_file("sha256_wrapper.cl", &src_size);

  const char *sources[]  = { src };
  const size_t lengths[] = { src_size };

  cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
  CHECK_CL(err, "clCreateProgramWithSource");

  // 如果需要 -I 等编译选项，在这里加
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

  if (err != CL_SUCCESS)
  {
    // 打印 build log
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *)malloc(log_size + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    log[log_size] = '\0';
    fprintf(stderr, "Build failed:\n%s\n", log);
    free(log);
    CHECK_CL(err, "clBuildProgram");
  }

  free(src);

  // 5. 创建 kernel
  cl_kernel kernel = clCreateKernel(program, "sha256_wrapper", &err);
  CHECK_CL(err, "clCreateKernel");

  // 6. 把 file_data 打包到 msgs (u32) + msg_lens (1条)
  const uint32_t len_bytes = (uint32_t)file_size;

  // stride 以 u32 为单位：向上取整
  const uint32_t msg_stride = (len_bytes + 3u) / 4u;

  uint32_t *msgs_host = (uint32_t *)calloc(msg_stride, sizeof(uint32_t));
  if (!msgs_host)
  {
    fprintf(stderr, "calloc msgs_host failed\n");
    exit(1);
  }

  // 按大端把 file_data 打成 u32：
  // 每4字节: (b0<<24)|(b1<<16)|(b2<<8)|b3；最后不足4字节用0填充
  for (uint32_t i = 0; i < msg_stride; i++)
  {
    uint32_t v = 0;
    size_t base = (size_t)i * 4;

    if (base + 0 < file_size) v |= (uint32_t)file_data[base + 0] << 24;
    if (base + 1 < file_size) v |= (uint32_t)file_data[base + 1] << 16;
    if (base + 2 < file_size) v |= (uint32_t)file_data[base + 2] <<  8;
    if (base + 3 < file_size) v |= (uint32_t)file_data[base + 3] <<  0;

    msgs_host[i] = v;
  }

  free(file_data); // 原始数据可以丢掉了

  uint32_t msg_lens_host[1];
  msg_lens_host[0] = len_bytes;

  uint32_t digests_host[8] = {0};

  // 7. 创建缓冲区
  cl_mem msgs_buf = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   msg_stride * sizeof(uint32_t),
                                   msgs_host,
                                   &err);
  CHECK_CL(err, "clCreateBuffer(msgs)");

  cl_mem lens_buf = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(msg_lens_host),
                                   msg_lens_host,
                                   &err);
  CHECK_CL(err, "clCreateBuffer(lens)");

  cl_mem digests_buf = clCreateBuffer(context,
                                      CL_MEM_WRITE_ONLY,
                                      sizeof(digests_host),
                                      NULL,
                                      &err);
  CHECK_CL(err, "clCreateBuffer(digests)");

  free(msgs_host);

  // 8. 设置 kernel 参数
  int arg = 0;
  CHECK_CL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &msgs_buf),     "clSetKernelArg(msgs)");
  CHECK_CL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &lens_buf),     "clSetKernelArg(lens)");
  CHECK_CL(clSetKernelArg(kernel, arg++, sizeof(uint32_t), &msg_stride), "clSetKernelArg(stride)");
  CHECK_CL(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &digests_buf),  "clSetKernelArg(digests)");

  // 9. 启动 kernel（1 条消息 -> 1 个 work-item）
  size_t global_work_size[1] = { 1 };
  CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                  global_work_size, NULL,
                                  0, NULL, NULL),
           "clEnqueueNDRangeKernel");

  CHECK_CL(clFinish(queue), "clFinish");

  // 10. 读回 digest
  CHECK_CL(clEnqueueReadBuffer(queue, digests_buf, CL_TRUE, 0,
                               sizeof(digests_host), digests_host,
                               0, NULL, NULL),
           "clEnqueueReadBuffer");

  // 11. 组装十六进制字符串
  char hex[64 + 1];
  char *p = hex;

  for (int i = 0; i < 8; i++)
  {
    uint32_t v = digests_host[i];
    // 大端输出
    sprintf(p, "%02x%02x%02x%02x",
            (unsigned)((v >> 24) & 0xff),
            (unsigned)((v >> 16) & 0xff),
            (unsigned)((v >>  8) & 0xff),
            (unsigned)((v      ) & 0xff));
    p += 8;
  }
  hex[64] = '\0';

  // 12. 写入输出文件
  FILE *fout = fopen(output_path, "wb");
  if (!fout)
  {
    perror(output_path);
    // 继续清理 OpenCL 资源
  }
  else
  {
    fprintf(fout, "%s\n", hex);
    fclose(fout);
  }

  // 同时在 stdout 打印一份，方便你验证
  printf("SHA256(%s) = %s\n", input_path, hex);

  // 13. 释放 OpenCL 资源
  clReleaseMemObject(msgs_buf);
  clReleaseMemObject(lens_buf);
  clReleaseMemObject(digests_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
