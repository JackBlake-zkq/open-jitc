#include <cstdlib>
#include <dlfcn.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <string>
#include <sstream>
#include <functional>
#include <atomic>
#include <iostream>
#include <fstream>
#include <set>
#include <fcntl.h>
#include <queue>
#include <set>
#include <thread>

#if TRACK_CUDA
#include <cuda_runtime.h>
#endif
#if TRACK_NCCL
#include <nccl.h>
#endif

#include "helpers.h"

#ifndef LIBTORCH_CUDA_PATH
#error "Path to libtorch_cuda.so not defined"
#endif



void *handle;
FILE *log_file;
char path[256];
bool allReduceHung = false;
std::multiset<long long> syncStartTimes;
const long long TIMEOUT = 1e10; // 10 seconds


long long currentTime() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    long long nanosecond_count = nanoseconds.count();
    return nanosecond_count;
}

void checkForHangs() {
    while(!allReduceHung) {
        if(syncStartTimes.empty()) {
            printf("syncStartTimes is empty\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        long long lastSyncTime = *syncStartTimes.begin();
        long long currTime = currentTime();
        printf("last sync time: %lld currentTime %lld\n", *syncStartTimes.begin(), currTime);
        fflush(stdout);
        if(lastSyncTime != 0 && currTime - lastSyncTime > TIMEOUT) {
            allReduceHung = true;
            fprintf(log_file, "Allreduce hang detected\n");
            fflush(log_file);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

}

__attribute__((constructor))
void my_init() {
    handle = dlopen(LIBTORCH_CUDA_PATH, RTLD_LAZY);
    auto cudaGetDevice = (cudaError_t (*)(int*)) dlsym(handle, "cudaGetDevice");
    int deviceID;
    cudaGetDevice(&deviceID);
    sprintf(path, "/tmp/interceptor_%d.log", deviceID);
    log_file = fopen(path, "w");
    if (log_file == NULL) {
        fprintf(stderr, "Error opening log file: %s\n", path);
        return;
    }
    std::thread tHangChecker(checkForHangs);
    tHangChecker.detach();
}



cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    printf("cudaStreamWaitEvent called\n");
    auto original_cudaStreamWaitEvent = (cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int))dlsym(handle, "cudaStreamWaitEvent");
    long long startTime  = currentTime();
    syncStartTimes.insert(startTime);
    cudaError_t result = original_cudaStreamWaitEvent(stream, event, flags);
    printf("cudaStreamWaitEvent done\n");
    syncStartTimes.erase(startTime);
    return result;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    printf("cudaEventRecord called\n");
    auto original_cudaEventRecord = (cudaError_t (*)(cudaEvent_t, cudaStream_t))dlsym(handle, "cudaEventRecord");
    long long startTime  = currentTime();
    syncStartTimes.insert(startTime);
    cudaError_t result = original_cudaEventRecord(event, stream);
    printf("cudaEventRecord done\n");
    syncStartTimes.erase(startTime);
    return result;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    auto original_cudaStreamCreate = (cudaError_t (*)(cudaStream_t*))dlsym(handle, "cudaStreamCreate");
    return original_cudaStreamCreate(pStream);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    auto original_cudaMemcpyAsync = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t))dlsym(handle, "cudaMemcpyAsync");
    return original_cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    auto original_cudaStreamSynchronize = (cudaError_t (*)(cudaStream_t))dlsym(handle, "cudaStreamSynchronize");
    return original_cudaStreamSynchronize(stream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    auto original_cudaStreamDestroy = (cudaError_t (*)(cudaStream_t))dlsym(handle, "cudaStreamDestroy");
    return original_cudaStreamDestroy(stream);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    auto original_cudaMemcpy = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind))dlsym(handle, "cudaMemcpy");
    if(allReduceHung) {
        // change to new CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(dst, src, count, kind, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    return original_cudaMemcpy(dst, src, count, kind);
}
