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

#define CREATE_HOOKED_CUDA_FUNCTION_BASE(return_type, func_name, args, arg_types, arg_names, PRE_RETURN_HOOK, ...) \
	return_type func_name args { \
		return_type (*original_##func_name) arg_types = NULL; \
		original_##func_name = (return_type (*) arg_types)dlsym(RTLD_NEXT, #func_name); \
		print_str(#func_name ":"); \
		EXPAND_ARGS arg_names \
		print_str("\n");\
	    cudaError_t result = original_##func_name arg_names; \
        PRE_RETURN_HOOK \
        return result; \
	}

// Then you specialize:

#define CREATE_HOOKED_CUDA_FUNCTION(return_type, func_name, args, arg_types, arg_names, ...) \
    CREATE_HOOKED_CUDA_FUNCTION_BASE(return_type, func_name, args, arg_types, arg_names, /* no pre-return hook */ , __VA_ARGS__)

void print_str(const char *str)
{
    printf("%s", str);
}

template<typename...> using void_t = void;

// Use to ensure variable is streamable
template<typename T, typename = void>
struct is_streamable : std::false_type {};

template<typename T>
struct is_streamable<
    T,
    void_t< decltype(std::declval<std::ostream&>() << std::declval<T>()) >
> : std::true_type {};

// Update printIfPrintable to ensure it works on all datatypes
template<typename T>
void printIfPrintable(const T& value) {
    std::ostringstream ss;
    if constexpr (is_streamable<T>::value) {
        // data types with <<operator>>
        ss << value;
    }
    else {
        // Print address otherwise
        ss << static_cast<const void*>(&value);
    }
    
    ss << ' ';
    printf("%s", ss.str().c_str());
}

void *handle;
// FILE *log_file;
std::ifstream * app_log_file;
bool useAltCudaStream = false;
char path[256];
int deviceID;
cudaStream_t newStream;
// std::multiset<long long> syncStartTimes;
// const long long TIMEOUT = 1e10; // 10 seconds


// long long currentTime() {
//     auto now = std::chrono::system_clock::now();
//     auto duration = now.time_since_epoch();
//     auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
//     long long nanosecond_count = nanoseconds.count();
//     return nanosecond_count;
// }



void checkAppLog() {
    auto origninal_cudaStreamCreateWithPriority = (cudaError_t (*) (cudaStream_t*, unsigned int, int))dlsym(handle, "cudaStreamCreateWithPriority");
    app_log_file = new std::ifstream(path);
    while(!app_log_file->is_open()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        app_log_file->open(path);
    }
    if (!app_log_file->is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    printf("Openned app log file %s\n", path);
    std::string line;
    while(!useAltCudaStream) {
        std::getline(*app_log_file, line);
        if(!line.empty()) {
            printf("Interceptor read line from app log: %s\n", line.c_str());
            fflush(stdout);
        }
        if (line.find("failed") != std::string::npos) {
            useAltCudaStream = true;
            origninal_cudaStreamCreateWithPriority(&newStream, cudaStreamNonBlocking, 999);
            printf("Using alterantive CUDA stream for mem copies from now on\n");
            // fprintf(log_file, "Detected hang in allReduce\n");
            // fflush(log_file);
            break;
        } else {
            if (app_log_file->eof()) {
                app_log_file->clear(); // Clear EOF flag
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait a bit for new data
            } else {
                std::cerr << "Error reading from log file" << std::endl;
                break;
            }
        }
    }

}

__attribute__((constructor))
void my_init() {
    printf("Tracer initialized\n");
    handle = dlopen(LIBTORCH_CUDA_PATH, RTLD_LAZY);
    auto cudaGetDevice = (cudaError_t (*)(int*)) dlsym(handle, "cudaGetDevice");
    cudaGetDevice(&deviceID);
    sprintf(path, "/tmp/app_%d.log", deviceID);
    // sprintf(path, "/tmp/interceptor_%d.log", deviceID);
    // log_file = fopen(path, "w");
    // if (log_file == NULL) {
    //     fprintf(stderr, "Error opening log file: %s\n", path);
    //     return;
    // }
    std::remove(path);

    std::thread tHangChecker(checkAppLog);
    tHangChecker.detach();
}



// cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
//     printf("cudaStreamWaitEvent called\n");
//     auto original_cudaStreamWaitEvent = (cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int))dlsym(handle, "cudaStreamWaitEvent");
//     long long startTime  = currentTime();
//     syncStartTimes.insert(startTime);
//     cudaError_t result = original_cudaStreamWaitEvent(stream, event, flags);
//     printf("cudaStreamWaitEvent done\n");
//     syncStartTimes.erase(startTime);
//     return result;
// }
// cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
//     printf("cudaEventRecord called\n");
//     auto original_cudaEventRecord = (cudaError_t (*)(cudaEvent_t, cudaStream_t))dlsym(handle, "cudaEventRecord");
//     long long startTime  = currentTime();
//     syncStartTimes.insert(startTime);
//     cudaError_t result = original_cudaEventRecord(event, stream);
//     printf("cudaEventRecord done\n");
//     syncStartTimes.erase(startTime);
//     return result;
// }

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    // printf("cudaStreamSynchronize called\n");
    auto original_cudaStreamSynchronize = (cudaError_t (*)(cudaStream_t))dlsym(handle, "cudaStreamSynchronize");
    if(useAltCudaStream) {
        // printf("Using alternative CUDA stream for sync\n");
        // change to new CUDA stream
        cudaError_t result = original_cudaStreamSynchronize(newStream);
        // printf("cudaStreamSynchronize done\n");
        return result;
    }
    return original_cudaStreamSynchronize(stream);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    // printf("cudaMemcpy called\n");
    auto original_cudaMemcpy = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind))dlsym(handle, "cudaMemcpy");
    if(useAltCudaStream) {
        // printf("Using alternative CUDA stream for memcpy\n");
        auto original_cudaMemcpyAsync = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t))dlsym(handle, "cudaMemcpyAsync");
        auto original_cudaStreamSynchronize = (cudaError_t (*)(cudaStream_t))dlsym(handle, "cudaStreamSynchronize");
        cudaError_t result;
        result = original_cudaMemcpyAsync(dst, src, count, kind, newStream);
        if (result != cudaSuccess) {
            printf("cudaMemcpyAsync failed: %s\n", cudaGetErrorString(result));
            return result;
        }
        result = original_cudaStreamSynchronize(newStream);
        // printf("cudaMemcpy done\n");
        return result;
    }
    return original_cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    // printf("cudaMemcpyAsync called\n");
    auto original_cudaMemcpyAsync = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t))dlsym(handle, "cudaMemcpyAsync");
    if(useAltCudaStream) {
        // printf("Using alternative CUDA stream for memcpy async\n");
        // change to new CUDA stream
        cudaError_t result = original_cudaMemcpyAsync(dst, src, count, kind, newStream);
        // printf("cudaMemcpyAsync done\n");
        return result;

    }
    return original_cudaMemcpyAsync(dst, src, count, kind, stream);
}

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t,
//     cudaArrayGetInfo,
//     (cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array),
//     (cudaChannelFormatDesc*, cudaExtent*, unsigned int*, cudaArray_t),
//     (desc, extent, flags, array),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaArrayGetPlane, 
//     (cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx), 
//     (cudaArray_t*, cudaArray_t, unsigned int), 
//     (pPlaneArray, hArray, planeIdx),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaArrayGetSparseProperties, 
//     (cudaArraySparseProperties* sparseProperties, cudaArray_t array), 
//     (cudaArraySparseProperties*, cudaArray_t), 
//     (sparseProperties, array),

//     )


// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaFreeArray, 
//     (cudaArray_t array), 
//     (cudaArray_t), 
//     (array),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaFreeHost, 
//     (void* ptr), 
//     (void*), 
//     (ptr),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaFreeMipmappedArray, 
//     (cudaMipmappedArray_t mipmappedArray), 
//     (cudaMipmappedArray_t), 
//     (mipmappedArray),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaGetMipmappedArrayLevel, 
//     (cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level), 
//     (cudaArray_t*, cudaMipmappedArray_const_t, unsigned int), 
//     (levelArray, mipmappedArray, level),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaGetSymbolAddress, 
//     (void** devPtr, const void* symbol), 
//     (void**, const void*), 
//     (devPtr, symbol),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaGetSymbolSize, 
//     (size_t* size, const void* symbol), 
//     (size_t*, const void*), 
//     (size, symbol),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaHostAlloc, 
//     (void** pHost, size_t size, unsigned int flags), 
//     (void**, size_t, unsigned int), 
//     (pHost, size, flags),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t,
//     cudaHostGetDevicePointer,
//     (void** pDevice, void* pHost, unsigned int flags),
//     (void**, void*, unsigned int),
//     (pDevice, pHost, flags),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaHostGetFlags, 
//     (unsigned int* pFlags, void* pHost), 
//     (unsigned int*, void*), 
//     (pFlags, pHost),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaHostRegister, 
//     (void* ptr, size_t size, unsigned int flags), 
//     (void*, size_t, unsigned int), 
//     (ptr, size, flags),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaHostUnregister, 
//     (void* ptr), 
//     (void*), 
//     (ptr),

//     )



// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaMalloc3D, 
//     (cudaPitchedPtr* pitchedDevPtr, cudaExtent extent), 
//     (cudaPitchedPtr*, cudaExtent), 
//     (pitchedDevPtr, extent),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
//     cudaError_t, 
//     cudaMalloc3DArray, 
//     (cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags), 
//     (cudaArray_t*, const cudaChannelFormatDesc*, cudaExtent, unsigned int), 
//     (array, desc, extent, flags),

//     )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMallocArray, 
// (cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags), 
// (cudaArray_t*, const cudaChannelFormatDesc*, size_t, size_t, unsigned int), 
// (array, desc, width, height, flags),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMallocHost, 
// (void** ptr, size_t size), 
// (void**, size_t), 
// (ptr, size),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMallocManaged, 
// (void** devPtr, size_t size, unsigned int flags), 
// (void**, size_t, unsigned int), 
// (devPtr, size, flags),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMallocMipmappedArray, 
// (cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags), 
// (cudaMipmappedArray_t*, const cudaChannelFormatDesc*, cudaExtent, unsigned int, unsigned int), 
// (mipmappedArray, desc, extent, numLevels, flags),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMallocPitch, 
// (void** devPtr, size_t* pitch, size_t width, size_t height), 
// (void**, size_t*, size_t, size_t), 
// (devPtr, pitch, width, height),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemAdvise, 
// (const void* devPtr, size_t count, cudaMemoryAdvise advice, int device), 
// (const void*, size_t, cudaMemoryAdvise, int), 
// (devPtr, count, advice, device),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemAdvise_v2, 
// (const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location), 
// (const void*, size_t, cudaMemoryAdvise, cudaMemLocation), 
// (devPtr, count, advice, location),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemGetInfo, 
// (size_t* free, size_t* total), 
// (size_t*, size_t*), 
// (free, total),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemPrefetchAsync, 
// (const void* devPtr, size_t count, int dstDevice, cudaStream_t stream), 
// (const void*, size_t, int, cudaStream_t), 
// (devPtr, count, dstDevice, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemRangeGetAttribute, 
// (void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count), 
// (void*, size_t, cudaMemRangeAttribute, const void*, size_t), 
// (data, dataSize, attribute, devPtr, count),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemRangeGetAttributes, 
// (void** data, size_t* dataSizes, cudaMemRangeAttribute** attributes, size_t numAttributes, const void* devPtr, size_t count), 
// (void**, size_t*, cudaMemRangeAttribute**, size_t, const void*, size_t), 
// (data, dataSizes, attributes, numAttributes, devPtr, count),

// )


// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy2D, 
// (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind), 
// (void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind), 
// (dst, dpitch, src, spitch, width, height, kind),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy2DArrayToArray, 
// (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind), 
// (cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind), 
// (dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy2DAsync, 
// (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), 
// (void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
// (dst, dpitch, src, spitch, width, height, kind, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy2DFromArray, 
// (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind), 
// (void*, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind), 
// (dst, dpitch, src, wOffset, hOffset, width, height, kind),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy2DFromArrayAsync, 
// (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), 
// (void*, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
// (dst, dpitch, src, wOffset, hOffset, width, height, kind, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy2DToArrayAsync, 
// (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), 
// (cudaArray_t, size_t, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
// (dst, wOffset, hOffset, src, spitch, width, height, kind, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy3D, 
// (const cudaMemcpy3DParms* p), 
// (const cudaMemcpy3DParms*), 
// (p),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy3DAsync, 
// (const cudaMemcpy3DParms* p, cudaStream_t stream), 
// (const cudaMemcpy3DParms*, cudaStream_t), 
// (p, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy3DPeer, 
// (const cudaMemcpy3DPeerParms* p), 
// (const cudaMemcpy3DPeerParms*), 
// (p),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpy3DPeerAsync, 
// (const cudaMemcpy3DPeerParms* p, cudaStream_t stream), 
// (const cudaMemcpy3DPeerParms*, cudaStream_t), 
// (p, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpyFromSymbol, 
// (void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind), 
// (void*, const void*, size_t, size_t, cudaMemcpyKind), 
// (dst, symbol, count, offset, kind),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpyFromSymbolAsync, 
// (void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), 
// (void*, const void*, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
// (dst, symbol, count, offset, kind, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpyPeer, 
// (void* dst, int dstDevice, const void* src, int srcDevice, size_t count), 
// (void*, int, const void*, int, size_t), 
// (dst, dstDevice, src, srcDevice, count),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpyPeerAsync, 
// (void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream), 
// (void*, int, const void*, int, size_t, cudaStream_t), 
// (dst, dstDevice, src, srcDevice, count, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpyToSymbol, 
// (const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind), 
// (const void*, const void*, size_t, size_t, cudaMemcpyKind), 
// (symbol, src, count, offset, kind),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemcpyToSymbolAsync, 
// (const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), 
// (const void*, const void*, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
// (symbol, src, count, offset, kind, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemset, 
// (void* devPtr, int value, size_t count), 
// (void*, int, size_t), 
// (devPtr, value, count),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemset2D, 
// (void* devPtr, size_t pitch, int value, size_t width, size_t height), 
// (void*, size_t, int, size_t, size_t), 
// (devPtr, pitch, value, width, height),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemset2DAsync, 
// (void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream), 
// (void*, size_t, int, size_t, size_t, cudaStream_t), 
// (devPtr, pitch, value, width, height, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemset3D, 
// (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent), 
// (cudaPitchedPtr, int, cudaExtent), 
// (pitchedDevPtr, value, extent),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMemset3DAsync, 
// (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream), 
// (cudaPitchedPtr, int, cudaExtent, cudaStream_t), 
// (pitchedDevPtr, value, extent, stream),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMipmappedArrayGetMemoryRequirements, 
// (cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device), 
// (cudaArrayMemoryRequirements*, cudaMipmappedArray_t, int), 
// (memoryRequirements, mipmap, device),

// )

// CREATE_HOOKED_CUDA_FUNCTION(
// cudaError_t, 
// cudaMipmappedArrayGetSparseProperties, 
// (cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap), 
// (cudaArraySparseProperties*, cudaMipmappedArray_t), 
// (sparseProperties, mipmap),
// )