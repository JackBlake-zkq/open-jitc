#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <time.h>
#include <nccl.h>


const char *log_file_path = "transparent_jitc.log";
static FILE *log_file = NULL;

__attribute__((constructor))
void init_logger() {
  log_file = fopen(log_file_path, "w");
  if (!log_file) {
    fprintf(stderr, "Failed to open log file!\n");
    exit(1);
  }
  fprintf(log_file, "[CUDA LOGGER] Initialized\n");
  fflush(log_file);
}

__attribute__((destructor))
void shutdown_logger() {
  if (log_file) {
    fprintf(log_file, "[CUDA LOGGER] Shutting down\n");
    fclose(log_file);
  }
}

void log_timestamp() {
  time_t now = time(nullptr);
  struct tm *t = localtime(&now);
  fprintf(log_file, "[%02d:%02d:%02d] ", t->tm_hour, t->tm_min, t->tm_sec);
}

// CUDA Logging

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  static cudaError_t (*real_cudaMalloc)(void **, size_t) = nullptr;
  if (!real_cudaMalloc)
    real_cudaMalloc = (cudaError_t (*)(void **, size_t)) dlsym(RTLD_NEXT, "cudaMalloc");

  cudaError_t result = real_cudaMalloc(devPtr, size);
  fprintf(log_file, "[cudaMalloc] size=%zu -> %p, result=%d\n", size, *devPtr, result);
  fflush(log_file);
  return result;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
  static cudaError_t (*real_cudaMemcpy)(void *, const void *, size_t, cudaMemcpyKind) = nullptr;
  if (!real_cudaMemcpy)
    real_cudaMemcpy = (cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind))
      dlsym(RTLD_NEXT, "cudaMemcpy");

  cudaError_t result = real_cudaMemcpy(dst, src, count, kind);
  fprintf(log_file, "[cudaMemcpy] size=%zu, kind=%d -> result=%d\n", count, kind, result);
  fflush(log_file);
  return result;
}

cudaError_t cudaFree(void *devPtr) {
  static auto real = (cudaError_t (*)(void *)) dlsym(RTLD_NEXT, "cudaFree");
  cudaError_t result = real(devPtr);
  log_timestamp();
  fprintf(log_file, "cudaFree(%p) -> result=%d\n", devPtr, result);
  fflush(log_file);
  return result;
}

cudaError_t cudaGetLastError(void) {
  static auto real = (cudaError_t (*)()) dlsym(RTLD_NEXT, "cudaGetLastError");
  cudaError_t result = real();
  log_timestamp();
  fprintf(log_file, "cudaGetLastError() -> result=%d\n", result);
  fflush(log_file);
  return result;
}

cudaError_t cudaDeviceSynchronize(void) {
  static auto real = (cudaError_t (*)()) dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
  log_timestamp();
  fprintf(log_file, "cudaDeviceSynchronize()\n");
  cudaError_t result = real();
  fprintf(log_file, " -> result=%d\n", result);
  fflush(log_file);
  return result;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
  static auto real = (cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t))
    dlsym(RTLD_NEXT, "cudaLaunchKernel");

  log_timestamp();
  fprintf(log_file,
          "cudaLaunchKernel(func=%p, grid=(%d,%d,%d), block=(%d,%d,%d), sharedMem=%zu, stream=%p)\n",
          func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, (void *)stream);

  cudaError_t result = real(func, gridDim, blockDim, args, sharedMem, stream);
  fprintf(log_file, " -> result=%d\n", result);
  fflush(log_file);
  return result;
}

//// NCCL Logging
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream) {
    
    static auto real = (ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)) dlsym(RTLD_DEFAULT, "ncclAllReduce");

    if (!real) {
      fprintf(log_file, "[NCCL LOGGER] Failed to load real ncclAllReduce: %s\n", dlerror());
    }

    if (!comm || !stream) {
      fprintf(stderr, "[NCCL LOGGER] Skipping ncclAllReduce due to null comm or stream\n");
      return ncclInvalidArgument;
    }

    if (log_file) {
        fprintf(log_file, "[NCCL] ncclAllReduce called with count: %zu\n", count);
        fclose(log_file);
    }

    ncclResult_t result = real(sendbuff, recvbuff, count, datatype, op, comm, stream);
    if (log_file) {
        fprintf(log_file, "[NCCL] ncclAllReduce result: %d\n", result);
        fflush(log_file);
    }
    char * error = ncclGetLastError(comm);
    if (error) {
        fprintf(log_file, "[NCCL] ncclAllReduce error: %s\n", error);
        fflush(log_file);
    }
    return result;
}
