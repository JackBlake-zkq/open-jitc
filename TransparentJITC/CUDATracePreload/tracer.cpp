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

#if TRACK_CUDA
#include <cuda_runtime.h>
#endif
#if TRACK_NCCL
#include <nccl.h>
#endif

#include "helpers.h"

#ifndef LOG_PATH
#define LOG_PATH "log/"
#endif

#ifndef LIBTORCH_CUDA_PATH
#error "Path to libtorch_cuda.so not defined"
#endif

#if TRACK_NCCL
#define CREATE_HOOKED_NCCL_FUNCTION(return_type, func_name, args, arg_types, arg_names, ...) \
	return_type func_name args { \
		return_type (*original_##func_name) arg_types = NULL; \
		original_##func_name = (return_type (*) arg_types)dlsym(handle, #func_name); \
		int deviceID; \
		cudaGetDevice(&deviceID); \
	    print_str(#func_name ":"); \
        checkAppLog(); \
		EXPAND_ARGS arg_names \
		print_str("\n");\
		return original_##func_name arg_names; \
	}
#endif

#if TRACK_CUDA

#define CREATE_HOOKED_CUDA_FUNCTION_BASE(return_type, func_name, args, arg_types, arg_names, PRE_RETURN_HOOK, ...) \
	return_type func_name args { \
		return_type (*original_##func_name) arg_types = NULL; \
		original_##func_name = (return_type (*) arg_types)dlsym(RTLD_NEXT, #func_name); \
		int deviceID; \
		cudaGetDevice(&deviceID); \
		print_str(#func_name ":"); \
		EXPAND_ARGS arg_names \
		print_str("\n");\
        checkAppLog(); \
	cudaError_t result = original_##func_name arg_names; \
        if(isPreOptStepTrainsientError(result)) { \
            handlePreOptStepTransientError(); \
            return cudaSuccess; \
        } \
        PRE_RETURN_HOOK \
        return result; \
	}

// Then you specialize:

#define CREATE_HOOKED_CUDA_FUNCTION(return_type, func_name, args, arg_types, arg_names, ...) \
    CREATE_HOOKED_CUDA_FUNCTION_BASE(return_type, func_name, args, arg_types, arg_names, /* no pre-return hook */ , __VA_ARGS__)

#define CREATE_HOOKED_CUDA_FUNCTION_MALLOC(args, arg_types, arg_names, ...) \
    CREATE_HOOKED_CUDA_FUNCTION_BASE(cudaError_t, cudaMalloc, args, arg_types, arg_names, handleCudaMalloc(devPtr, size);, __VA_ARGS__)

#define CREATE_HOOKED_CUDA_FUNCTION_FREE(args, arg_types, arg_names, ...) \
    CREATE_HOOKED_CUDA_FUNCTION_BASE(cudaError_t, cudaFree, args, arg_types, arg_names, handleCudaFree(devPtr);, __VA_ARGS__)

#endif


void *handle;
FILE *log_file;
FILE *app_log_file;
char path[256];
bool inBatch = false;
bool inOptStep = false;
struct Allocation {
    uintptr_t addr;
    size_t size;
    bool operator<(const Allocation& other) const {
        return addr < other.addr;
    }
};
std::set<Allocation> startOfBatchState;
std::set<Allocation> withinBatchState;
bool noopAll = false;


__attribute__((constructor))
void my_init() {
    handle = dlopen(LIBTORCH_CUDA_PATH, RTLD_LAZY);
    auto cudaGetDevice = (cudaError_t (*)(int*)) dlsym(handle, "cudaGetDevice");
    int deviceID;
    cudaGetDevice(&deviceID);
    sprintf(path, LOG_PATH "interceptor_%d.log", deviceID);
    log_file = fopen(path, "w");
    if (log_file == NULL) {
        fprintf(stderr, "Error opening log file: %s\n", path);
        return;
    }

    char app_log_path[256];
    sprintf(app_log_path, "/tmp/app_%d.log", deviceID);
    app_log_file = fopen(app_log_path, "r");
    if (app_log_file == NULL) {
        fprintf(stderr, "Error opening app log file: %s\n", app_log_path);
        return;
    }

    
}

void print_str(const char *str)
{
	fprintf(log_file, "%s", str);
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
    fprintf(log_file, "%s", ss.str().c_str());
}

bool isPreOptStepTrainsientError(cudaError_t result) {
    if(inOptStep) {
        return false;
    }
    return rand() < 0.001;
}

void replayLog() {
    // std::ifstream slog_file(path);
    // std::string line;
    // while (std::getline(slog_file, line)) {
    //     std::string funcName = line.substr(0,line.find_first_of(":"));
    //     std::string args = line.substr(line.find_first_of(":")+1);
    //     // func = dlsym(RTLD_NEXT, funcName);

    // }
    // slog_file.close();
    fprintf(log_file, "Replay log\n");
    printf("Replay log\n");
}

void handlePreOptStepTransientError() {
    // fprintf(log_file, "Transient error occurred\n");
    // fflush(log_file);
    auto realCudaFree = (cudaError_t (*)(void *)) dlsym(handle, "cudaFree");
    for(auto it = withinBatchState.begin(); it != withinBatchState.end(); ++it) {
        uintptr_t addr = it->addr;
        size_t size = it->size;
        void *ptr = reinterpret_cast<void*>(addr);
        realCudaFree(ptr);
        withinBatchState.erase(it);
    }
    replayLog();
}


void handleCudaMalloc(void** devPtr, size_t size) {
    std::set<Allocation> * targetSet = inBatch ? &withinBatchState : &startOfBatchState;
    Allocation alloc;
    alloc.addr = reinterpret_cast<uintptr_t>(*devPtr);
    alloc.size = size;
    targetSet->insert(alloc);
}
void handleCudaFree(void* devPtr) {
    std::set<Allocation> * targetSet = inBatch ? &withinBatchState : &startOfBatchState;
    Allocation alloc;
    alloc.addr = reinterpret_cast<uintptr_t>(devPtr);
    auto it = targetSet->find(alloc);
    if (it != targetSet->end()) {
        targetSet->erase(it);
    }
}

void clearLog() {
    fclose(log_file);
    log_file = fopen(path, "w");
    if (log_file == NULL) {
        fprintf(stderr, "Error opening log file: %s\n", path);
        return;
    }
}

void checkAppLog() {
    char line[256];

    // Ensure the file pointer is valid
    if (app_log_file == NULL) {
        fprintf(stderr, "Error: Log file is not open.\n");
        return;
    }

    // Set the file descriptor to non-blocking mode
    int fd = fileno(app_log_file);
    if (fd == -1) {
        fprintf(stderr, "Error: Invalid file descriptor.\n");
        return;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl(F_GETFL)");
        return;
    }

    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl(F_SETFL)");
        return;
    }

    while (1) {
        // Set up select() to check for file readiness
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(fd, &readfds);

        struct timeval timeout = {0, 0};  // No wait, non-blocking
        int result = select(fd + 1, &readfds, NULL, NULL, &timeout);

        if (result == -1) {
            return;
        }

        if (result > 0) {  // File is ready to read
            if (fgets(line, sizeof(line), app_log_file)) {
                // Check the content of the line
                if (strstr(line, "Batch started") != NULL) {
                    printf("Batch started\n");
                    inOptStep = false;
                    inBatch = true;
                    clearLog();
                } else if (strstr(line, "Optimizer step") != NULL) {
                    printf("Optimizer step\n");
                    inOptStep = true;
                }
            } else {
                // If fgets fails, break the loop (or handle as necessary)
                if (feof(app_log_file)) {
                    break;  // End of file reached
                }
                perror("fgets()");
                return;
            }
        }

        // Sleep briefly to avoid high CPU usage in the loop
        usleep(100);  // 0.1 ms sleep
    }
}


#if TRACK_CUDA
CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaMemcpy,
		(void* dst, const void* src, size_t count, cudaMemcpyKind kind),
		(void *, const void*, size_t, cudaMemcpyKind),
		(dst, src, count, kind),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t,
		cudaMemcpyAsync,
		(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str),
		(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t),
		(dst, src, count, kind, str),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t,
		cudaArrayGetInfo,
		(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array),
		(cudaChannelFormatDesc*, cudaExtent*, unsigned int*, cudaArray_t),
		(desc, extent, flags, array),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaArrayGetPlane, 
		(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx), 
		(cudaArray_t*, cudaArray_t, unsigned int), 
		(pPlaneArray, hArray, planeIdx),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaArrayGetSparseProperties, 
		(cudaArraySparseProperties* sparseProperties, cudaArray_t array), 
		(cudaArraySparseProperties*, cudaArray_t), 
		(sparseProperties, array),

		)

CREATE_HOOKED_CUDA_FUNCTION_FREE(
		(void* devPtr), 
		(void*), 
		(devPtr),
		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaFreeArray, 
		(cudaArray_t array), 
		(cudaArray_t), 
		(array),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaFreeHost, 
		(void* ptr), 
		(void*), 
		(ptr),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaFreeMipmappedArray, 
		(cudaMipmappedArray_t mipmappedArray), 
		(cudaMipmappedArray_t), 
		(mipmappedArray),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaGetMipmappedArrayLevel, 
		(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level), 
		(cudaArray_t*, cudaMipmappedArray_const_t, unsigned int), 
		(levelArray, mipmappedArray, level),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaGetSymbolAddress, 
		(void** devPtr, const void* symbol), 
		(void**, const void*), 
		(devPtr, symbol),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaGetSymbolSize, 
		(size_t* size, const void* symbol), 
		(size_t*, const void*), 
		(size, symbol),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaHostAlloc, 
		(void** pHost, size_t size, unsigned int flags), 
		(void**, size_t, unsigned int), 
		(pHost, size, flags),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t,
		cudaHostGetDevicePointer,
		(void** pDevice, void* pHost, unsigned int flags),
		(void**, void*, unsigned int),
		(pDevice, pHost, flags),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaHostGetFlags, 
		(unsigned int* pFlags, void* pHost), 
		(unsigned int*, void*), 
		(pFlags, pHost),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaHostRegister, 
		(void* ptr, size_t size, unsigned int flags), 
		(void*, size_t, unsigned int), 
		(ptr, size, flags),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaHostUnregister, 
		(void* ptr), 
		(void*), 
		(ptr),

		)

std::atomic<long> total_mb_allocated(0);

CREATE_HOOKED_CUDA_FUNCTION_MALLOC(
		(void** devPtr, size_t size), 
		(void**, size_t), 
		(devPtr, size),
		[=]() { total_mb_allocated.fetch_add(size / 1000000, std::memory_order_relaxed); } 
		)


CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaMalloc3D, 
		(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent), 
		(cudaPitchedPtr*, cudaExtent), 
		(pitchedDevPtr, extent),

		)

CREATE_HOOKED_CUDA_FUNCTION(
		cudaError_t, 
		cudaMalloc3DArray, 
		(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags), 
		(cudaArray_t*, const cudaChannelFormatDesc*, cudaExtent, unsigned int), 
		(array, desc, extent, flags),

		)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMallocArray, 
    (cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags), 
    (cudaArray_t*, const cudaChannelFormatDesc*, size_t, size_t, unsigned int), 
    (array, desc, width, height, flags),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMallocHost, 
    (void** ptr, size_t size), 
    (void**, size_t), 
    (ptr, size),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMallocManaged, 
    (void** devPtr, size_t size, unsigned int flags), 
    (void**, size_t, unsigned int), 
    (devPtr, size, flags),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMallocMipmappedArray, 
    (cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags), 
    (cudaMipmappedArray_t*, const cudaChannelFormatDesc*, cudaExtent, unsigned int, unsigned int), 
    (mipmappedArray, desc, extent, numLevels, flags),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMallocPitch, 
    (void** devPtr, size_t* pitch, size_t width, size_t height), 
    (void**, size_t*, size_t, size_t), 
    (devPtr, pitch, width, height),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemAdvise, 
    (const void* devPtr, size_t count, cudaMemoryAdvise advice, int device), 
    (const void*, size_t, cudaMemoryAdvise, int), 
    (devPtr, count, advice, device),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemAdvise_v2, 
    (const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location), 
    (const void*, size_t, cudaMemoryAdvise, cudaMemLocation), 
    (devPtr, count, advice, location),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemGetInfo, 
    (size_t* free, size_t* total), 
    (size_t*, size_t*), 
    (free, total),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemPrefetchAsync, 
    (const void* devPtr, size_t count, int dstDevice, cudaStream_t stream), 
    (const void*, size_t, int, cudaStream_t), 
    (devPtr, count, dstDevice, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemRangeGetAttribute, 
    (void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count), 
    (void*, size_t, cudaMemRangeAttribute, const void*, size_t), 
    (data, dataSize, attribute, devPtr, count),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemRangeGetAttributes, 
    (void** data, size_t* dataSizes, cudaMemRangeAttribute** attributes, size_t numAttributes, const void* devPtr, size_t count), 
    (void**, size_t*, cudaMemRangeAttribute**, size_t, const void*, size_t), 
    (data, dataSizes, attributes, numAttributes, devPtr, count),
    
)


CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy2D, 
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind), 
    (void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind), 
    (dst, dpitch, src, spitch, width, height, kind),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy2DArrayToArray, 
    (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind), 
    (cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind), 
    (dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy2DAsync, 
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), 
    (void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
    (dst, dpitch, src, spitch, width, height, kind, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy2DFromArray, 
    (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind), 
    (void*, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind), 
    (dst, dpitch, src, wOffset, hOffset, width, height, kind),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy2DFromArrayAsync, 
    (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), 
    (void*, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
    (dst, dpitch, src, wOffset, hOffset, width, height, kind, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy2DToArrayAsync, 
    (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream), 
    (cudaArray_t, size_t, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
    (dst, wOffset, hOffset, src, spitch, width, height, kind, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy3D, 
    (const cudaMemcpy3DParms* p), 
    (const cudaMemcpy3DParms*), 
    (p),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy3DAsync, 
    (const cudaMemcpy3DParms* p, cudaStream_t stream), 
    (const cudaMemcpy3DParms*, cudaStream_t), 
    (p, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy3DPeer, 
    (const cudaMemcpy3DPeerParms* p), 
    (const cudaMemcpy3DPeerParms*), 
    (p),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpy3DPeerAsync, 
    (const cudaMemcpy3DPeerParms* p, cudaStream_t stream), 
    (const cudaMemcpy3DPeerParms*, cudaStream_t), 
    (p, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpyFromSymbol, 
    (void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind), 
    (void*, const void*, size_t, size_t, cudaMemcpyKind), 
    (dst, symbol, count, offset, kind),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpyFromSymbolAsync, 
    (void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), 
    (void*, const void*, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
    (dst, symbol, count, offset, kind, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpyPeer, 
    (void* dst, int dstDevice, const void* src, int srcDevice, size_t count), 
    (void*, int, const void*, int, size_t), 
    (dst, dstDevice, src, srcDevice, count),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpyPeerAsync, 
    (void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream), 
    (void*, int, const void*, int, size_t, cudaStream_t), 
    (dst, dstDevice, src, srcDevice, count, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpyToSymbol, 
    (const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind), 
    (const void*, const void*, size_t, size_t, cudaMemcpyKind), 
    (symbol, src, count, offset, kind),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemcpyToSymbolAsync, 
    (const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream), 
    (const void*, const void*, size_t, size_t, cudaMemcpyKind, cudaStream_t), 
    (symbol, src, count, offset, kind, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemset, 
    (void* devPtr, int value, size_t count), 
    (void*, int, size_t), 
    (devPtr, value, count),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemset2D, 
    (void* devPtr, size_t pitch, int value, size_t width, size_t height), 
    (void*, size_t, int, size_t, size_t), 
    (devPtr, pitch, value, width, height),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemset2DAsync, 
    (void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream), 
    (void*, size_t, int, size_t, size_t, cudaStream_t), 
    (devPtr, pitch, value, width, height, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemset3D, 
    (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent), 
    (cudaPitchedPtr, int, cudaExtent), 
    (pitchedDevPtr, value, extent),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMemset3DAsync, 
    (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream), 
    (cudaPitchedPtr, int, cudaExtent, cudaStream_t), 
    (pitchedDevPtr, value, extent, stream),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMipmappedArrayGetMemoryRequirements, 
    (cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device), 
    (cudaArrayMemoryRequirements*, cudaMipmappedArray_t, int), 
    (memoryRequirements, mipmap, device),
    
)

CREATE_HOOKED_CUDA_FUNCTION(
    cudaError_t, 
    cudaMipmappedArrayGetSparseProperties, 
    (cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap), 
    (cudaArraySparseProperties*, cudaMipmappedArray_t), 
    (sparseProperties, mipmap),
    
)
#endif


#if TRACK_NCCL

std::atomic<long> total_allreduce_count(0);

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclAllReduce, 
    (const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream), 
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t), 
    (sendbuff, recvbuff, count, datatype, op, comm, stream),
    
    [=]() { total_allreduce_count.fetch_add(count, std::memory_order_relaxed); }
)

std::atomic<long> total_broadcast_count(0);

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclBroadcast, 
    (const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream), 
    (const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t), 
    (sendbuff, recvbuff, count, datatype, root, comm, stream),
    
    [=]() { total_broadcast_count.fetch_add(count, std::memory_order_relaxed); }
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclBcast, 
    (void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream), 
    (void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t), 
    (buff, count, datatype, root, comm, stream)
)

std::atomic<long> total_reduce_count(0);
CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclReduce, 
    (const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream), 
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t), 
    (sendbuff, recvbuff, count, datatype, op, root, comm, stream),
    
    [=]() { total_reduce_count.fetch_add(count, std::memory_order_relaxed); }
)

std::atomic<long> total_allgather_count(0);
CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclAllGather, 
    (const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream), 
    (const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t), 
    (sendbuff, recvbuff, sendcount, datatype, comm, stream),
    
    [=]() { total_allgather_count.fetch_add(sendcount, std::memory_order_relaxed); }
)

std::atomic<long> total_reducescatter_count(0);
CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclReduceScatter, 
    (const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream), 
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t), 
    (sendbuff, recvbuff, recvcount, datatype, op, comm, stream),
    
    [=]() { total_reducescatter_count.fetch_add(recvcount, std::memory_order_relaxed); }
)

std::atomic<long> total_ncclmemalloc_count(0);
CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclMemAlloc, 
    (void **ptr, size_t size), 
    (void**, size_t), 
    (ptr, size),
    
    [=]() { total_ncclmemalloc_count.fetch_add(size, std::memory_order_relaxed); }
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclMemFree, 
    (void *ptr), 
    (void*), 
    (ptr)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclCommDeregister, 
    (const ncclComm_t comm, void* handle), 
    (ncclComm_t, void*), 
    (comm, handle)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclCommRegister, 
    (const ncclComm_t comm, void* buff, size_t size, void** handle), 
    (ncclComm_t, void*, size_t, void**), 
    (comm, buff, size, handle)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclCommUserRank, 
    (const ncclComm_t comm, int* rank), 
    (ncclComm_t, int*), 
    (comm, rank)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclCommCuDevice, 
    (const ncclComm_t comm, int* device), 
    (ncclComm_t, int*), 
    (comm, device)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclCommCount, 
    (const ncclComm_t comm, int* count), 
    (ncclComm_t, int*), 
    (comm, count)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclCommInitAll, 
    (ncclComm_t* comms, int ndev, const int* devlist), 
    (ncclComm_t*, int, const int*), 
    (comms, ndev, devlist)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclSend, 
    (const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream), 
    (const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t), 
    (sendbuff, count, datatype, peer, comm, stream)
)

CREATE_HOOKED_NCCL_FUNCTION(
    ncclResult_t, 
    ncclRecv, 
    (void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream), 
    (void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t), 
    (recvbuff, count, datatype, peer, comm, stream)
)
#endif

/* We use this destructor to print results */
void __attribute__((destructor)) report() {
	print_str("### Report ###\n");
#if TRACK_CUDA
	fprintf(log_file, "Total cudaMalloc: %ld MB\n", total_mb_allocated.load(std::memory_order_relaxed));
#endif
#if TRACK_NCCL
	fprintf(log_file, "Total ncclAllReduce count: %ld bytes\n", total_allreduce_count.load(std::memory_order_relaxed));
	fprintf(log_file, "Total broadcast count: %ld bytes\n", total_broadcast_count.load(std::memory_order_relaxed));
	fprintf(log_file, "Total reduce count: %ld bytes\n", total_reduce_count.load(std::memory_order_relaxed));
	fprintf(log_file, "Total allgather count: %ld bytes\n", total_allgather_count.load(std::memory_order_relaxed));
	fprintf(log_file, "Total reducescatter count: %ld bytes\n", total_reducescatter_count.load(std::memory_order_relaxed));
	fprintf(log_file, "Total ncclmemalloc count: %ld bytes\n", total_ncclmemalloc_count.load(std::memory_order_relaxed));
#endif
}
