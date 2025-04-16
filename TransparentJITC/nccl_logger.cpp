#include <nccl.h>
#include <dlfcn.h>
#include <stdio.h>

static ncclResult_t (*real_ncclAllReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = nullptr;

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream) {
    if (!real_ncclAllReduce) {
        real_ncclAllReduce = (decltype(real_ncclAllReduce)) dlsym(RTLD_NEXT, "ncclAllReduce");
    }

    FILE* log = fopen("nccl_trace.log", "a");
    if (log) {
        fprintf(log, "[NCCL] ncclAllReduce called with count: %zu\n", count);
        fclose(log);
    }

    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}
