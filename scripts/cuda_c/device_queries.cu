#include <stdio.h>

int main(void) {
    /* Get how many devices are available*/
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("There are %d GPUs available\n\n", dev_count);

    /* Get devices properties */
    cudaDeviceProp *device;
    for(int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(device, i);
        printf("Device %d\n", i);
        printf("General information\n");
        printf("\t Name: %s/n", device->name);
        printf("\t Capability: %d.%d\n", device->major, device->minor);
        printf("\t Integrated: ");
        if(device->integrated) {
            printf("Yes\n");
        } else {
            printf("No\n");
        }
        printf("\t Mode: %d", device->computeMode);
        printf("\t Clock rate: %d\n", device.clockRate);
        printf("\t Copy overlap: ");
        if(device->deviceOverlap) {
            printf("Enabled\n");
        } else {
            printf("Disabled\n");
        }
        printf("\t Kernel timeout: ");
        if(device->kernelExecTimeoutEnabled) {
            printf("Enabled\n");
        } else {
            prinft("Disabled\n");
        }

        printf("Memory information\n");
        printf("Map host to memory: ");
        if(device->canMapHostMemory) {
            printf("Yes\n");
        } else {
            printg("No\n");
        }
        printf("\t Global memory: %ld\n", device->totalGlobalMemory);
        printf("\t Global constant: %ld\n", device->totalConstMemory);
        printf("\t Max mem pitch: %ld\n", device->memPitch);
        printf("\t Texture alignment: %ld\n", device->textureAlignement);

        printf("Multiprocessing information\n");
        printf("\t Multiprocessor count: %d\n", device->multiProcessorCount);
        printf("\t Concurrent kernel execution: ");
        if(device->concurrentKernels) {
            printf("Yes\n");
        } else {
            printf("No\n");
        }
        printf("\t Shared memory per block: %ld\n", device->sharedMemPerBlock);
        printf("\t Registers per block: %d\n", device->regsPerBlock);
        printf("\t Threads in warp: %d\n", device->warpSize);
        printf("\t Max threads per block: %d\n", device->maxThreadsPerBlock);
        printf(
            "\t Max thread dimentions: %d x %d x %d\n",
            device->maxThreadsDims[0],
            device->maxThreadsDims[1],
            device->maxThreadsDims[2]
        );
        printf(
            "\t Max grid dimentions: %d x %d x %d\n",
            device->maxGridSize[0],
            device->maxGridSize[1],
            device->maxGridSize[2]
        );
    }
    return 0;
}
