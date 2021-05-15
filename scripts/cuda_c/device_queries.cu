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
        prtinf("General information\n")
        printf("Name: %s/n", device->name);
        printf("Capability: %d - %d\n", device->minor, device->major);
        printf("Clock rate: %d\n", device.clockRate);
        printf("Copy overlap:");
        if(device->deviceOverlap) {
            printf("Enabled\n");
        } else {
            printf("Disabled\n");
        }
        printf("Kernel timeout:");
        if(device->kernelExecTimeoutEnabled) {
            printf("Enabled\n");
        } else {
            prinft("Disabled\n");
        }
    }
    return 0;
}
