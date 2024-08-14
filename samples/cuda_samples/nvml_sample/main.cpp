#include <cuda_runtime_api.h>
#include <nvml.h>
#include <stdio.h>

static const char *convertToComputeModeString(nvmlComputeMode_t mode)
{
    switch (mode)
    {
    case NVML_COMPUTEMODE_DEFAULT:
        return "Default";
    case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
        return "Exclusive_Thread";
    case NVML_COMPUTEMODE_PROHIBITED:
        return "Prohibited";
    case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
        return "Exclusive Process";
    default:
        return "Unknown";
    }
}

int main(int argc, char *argv[])
{
    nvmlReturn_t result;
    unsigned int device_count, i;

    // First initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));

        printf("Press ENTER to continue...\n");
        getchar();
        return 1;
    }
    result = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != result)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        return -1;
    }
    printf("Found %u device%s\n\n", device_count, device_count != 1 ? "s" : "");

    printf("Listing devices:\n");
    for (i = 0; i < device_count; i++)
    {
        nvmlDevice_t  device;
        char          name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlPciInfo_t pci;

        // 通过设备句柄查询设备以执行操作
        // 除了使用 PCI 总线 ID 之外，还可以通过其他特性来查询设备句柄，例如：
        // nvmlDeviceGetHandleBySerial
        // nvmlDeviceGetHandleByPciBusId
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get handle for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get name of device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }

        // pci.busId 对于识别您在物理上正在与哪个设备进行通信非常有用
        // 使用 PCI 标识符，您还可以将 nvmlDevice 句柄匹配到 CUDA 设备。
        result = nvmlDeviceGetPciInfo(device, &pci);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get pci info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("%u. %s [%s]\n", i, name, pci.busId);

        // 查询pcie
        unsigned int pcie_gen;
        unsigned int pcie_width, pcie_speed, pcie_tx_throughput, pcie_rx_throughput;
        result = nvmlDeviceGetCurrPcieLinkGeneration(device, &pcie_gen);
        result = nvmlDeviceGetCurrPcieLinkWidth(device, &pcie_width);
        result = nvmlDeviceGetPcieSpeed(device, &pcie_speed);
        result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &pcie_tx_throughput);
        result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &pcie_rx_throughput);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get max pcie link width or speed for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    PCIe Gen: %u\n    PCIe Width: %u\n    PCIe Speed: %u Mbps\n", pcie_gen, pcie_width, pcie_speed);
        printf("    PCIe TX Throughput: %u MB/s\n    PCIe RX Throughput: %u MB/s\n", pcie_tx_throughput / 1024 / 1024,
               pcie_rx_throughput / 1024 / 1024);

        // 查询显存信息
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get memory info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Memory(NVML): %f MiB / %f MiB\n", memory.used / 1024.0 / 1024.0, memory.total / 1024.0 / 1024.0);
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        size_t used_memory = total_memory - free_memory;
        printf("    Memory(CUDA): %f MiB / %f MiB\n", used_memory / 1024.0 / 1024.0, total_memory / 1024.0 / 1024.0);

        // 查询显卡驱动版本
        char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        result = nvmlSystemGetDriverVersion(driver_version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get driver info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Driver Version: %s\n", driver_version);

        // 查询当前显卡频率
        unsigned int graph_clock, mem_clock;
        // result = nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, &clock); // 不支持
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graph_clock);
        result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get clock info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }

        printf("    Graphics Clock: %u MHz\n", graph_clock);
        printf("    Graphics Clock: %u MHz\n", mem_clock);

        // 查询最大显卡频率
        unsigned int max_graph_clock, max_mem_clock;
        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &max_graph_clock);
        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &max_mem_clock);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get max clock info for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Max Graphics Clock: %u MHz\n", max_graph_clock);
        printf("    Max Memory Clock: %u MHz\n", max_mem_clock);

        // 查看显卡风扇
        unsigned int fan_speed;
        result = nvmlDeviceGetFanSpeed(device, &fan_speed);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get fan speed for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Fan Speed: %u %%\n", fan_speed);

        // 查询显卡温度
        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get temperature for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Temperature: %u C\n", temp);

        // 查询利用率
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get utilization for device %u: %s\n", i, nvmlErrorString(result));
            return -1;
        }
        printf("    Utilization: GPU: %u%%, Memory: %u%%\n", utilization.gpu, utilization.memory);
    }
    return 0;
}
