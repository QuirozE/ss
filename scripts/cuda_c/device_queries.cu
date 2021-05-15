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
        printf("\t Name: %s\n", device->name);
        printf("\t Capability: %d.%d\n", device->major, device->minor);
        printf("\t Integrated: ");
        if(device->integrated) {
            printf("Yes\n");
        } else {
            printf("No\n");
        }
        printf("\t Mode: ");
	switch(device->computeMode) {
	    case 0:
                printf("Default\n");
		break;
	    case 1:
		printf("Exclusive\n");
		break;
	    case 2:
		printf("Prohibited\n");
		break;
	}
        printf("\t Clock rate: %d\n", device->clockRate);
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
            printf("Disabled\n");
        }

        printf("Memory information\n");
        printf("\t Map host to memory: ");
        if(device->canMapHostMemory) {
            printf("Yes\n");
        } else {
            printf("No\n");
        }
        printf("\t Global memory: %ld\n", device->totalGlobalMem);
        printf("\t Global constant: %ld\n", device->totalConstMem);
        printf("\t Max mem pitch: %ld\n", device->memPitch);
        printf("\t Texture alignment: %ld\n", device->textureAlignment);

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
            device->maxThreadsDim[0],
            device->maxThreadsDim[1],
            device->maxThreadsDim[2]
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
