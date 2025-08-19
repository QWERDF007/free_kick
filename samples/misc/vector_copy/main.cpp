#include "vector_utils.h"

#include <chrono>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    std::vector<std::vector<double>> tmp;
    std::cout << "vector size: " << tmp.size() << ", capacity: " << tmp.capacity() << std::endl;
    tmp.reserve(2);
    std::cout << "vector size: " << tmp.size() << ", capacity: " << tmp.capacity() << std::endl;
    tmp.push_back({1, 2, 3, 4, 5, 6});
    std::cout << "vector size: " << tmp.size() << ", capacity: " << tmp.capacity() << std::endl;
    tmp.push_back({1, 2, 3, 4, 5, 6});
    std::cout << "vector size: " << tmp.size() << ", capacity: " << tmp.capacity() << std::endl;
    tmp.push_back({1, 2, 3, 4, 5, 6});
    std::cout << "vector size: " << tmp.size() << ", capacity: " << tmp.capacity() << std::endl;
    const int M = 5000;
    const int N = 6;
    double    data[N * M];
    for (int i = 0; i < N * M; ++i)
    {
        data[i] = i;
    }

    auto             start1 = std::chrono::high_resolution_clock::now();
    std::vector<int> xx(M);
    for (int i = 0; i < M; ++i)
    {
        xx[i] = i;
    }
    auto end1  = std::chrono::high_resolution_clock::now();
    auto time1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
    std::cout << "construct + assign " << M << " elements, Total time: " << time1 << "ms" << std::endl;

    auto             start2 = std::chrono::high_resolution_clock::now();
    std::vector<int> xx2;
    xx2.reserve(M);
    for (int i = 0; i < M; ++i)
    {
        xx2.push_back(i);
    }
    auto end2  = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
    std::cout << "reserve + push " << M << " elements, Total time: " << time2 << "ms" << std::endl;

    double                           total_time = 0;
    std::vector<std::vector<double>> dst0;
    for (int i = 0; i < M; ++i)
    {
        auto                start = std::chrono::high_resolution_clock::now();
        std::vector<double> dst(data + i * N, data + (i + 1) * N);
        dst0.push_back(dst);
        auto end  = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += time;
    }
    std::cout << "push + construct " << N << " elements " << M << " times, Total time: " << total_time
              << "ms, Average time: " << total_time / M << "ms" << std::endl;

    total_time = 0;
    std::vector<std::vector<double>> dst1;
    for (int i = 0; i < M; ++i)
    {
        auto                start = std::chrono::high_resolution_clock::now();
        std::vector<double> dst;
        dst.insert(dst.end(), data + i * N, data + (i + 1) * N);
        dst1.insert(dst1.end(), dst);
        auto end  = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += time;
    }
    std::cout << "insert + insert " << N << " elements " << M << " times, Total time: " << total_time
              << "ms, Average time: " << total_time / M << "ms" << std::endl;

    total_time = 0;
    std::vector<std::vector<double>> dst2;
    for (int i = 0; i < M; ++i)
    {
        std::vector<double> dst;
        auto                start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < N; ++j)
        {
            dst.push_back(data[j]);
        }
        dst2.push_back(dst);
        auto end  = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += time;
    }
    std::cout << "push + push " << N << " elements " << M << " times, Total time: " << total_time
              << "ms, Average time: " << total_time / M << "ms" << std::endl;

    total_time = 0;
    std::vector<std::vector<double>> dst3;
    for (int i = 0; i < M; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        dst3.push_back(
            {data[i * N], data[i * N + 1], data[i * N + 2], data[i * N + 3], data[i * N + 4], data[i * N + 5]});
        auto end  = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += time;
    }
    std::cout << "push + init_list " << N << " elements " << M << " times, Total time: " << total_time
              << "ms, Average time: " << total_time / M << "ms" << std::endl;

    total_time = 0;
    std::vector<std::vector<double>> dst4;
    for (int i = 0; i < M; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        dst4.insert(dst4.end(),
                    {data[i * N], data[i * N + 1], data[i * N + 2], data[i * N + 3], data[i * N + 4], data[i * N + 5]});
        auto end  = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += time;
    }
    std::cout << "insert + init_list " << N << " elements " << M << " times, Total time: " << total_time
              << "ms, Average time: " << total_time / M << "ms" << std::endl;

    total_time                             = 0;
    auto                             start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> dst5(M, std::vector<double>());
    auto                             end  = std::chrono::high_resolution_clock::now();
    auto                             time = std::chrono::duration<double, std::milli>(end - start).count();
    total_time += time;
    for (int i = 0; i < M; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        dst5[i]    = {data[i * N], data[i * N + 1], data[i * N + 2], data[i * N + 3], data[i * N + 4], data[i * N + 5]};
        auto end   = std::chrono::high_resolution_clock::now();
        auto time  = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += time;
    }
    std::cout << "pre-alloc + init_list " << N << " elements " << M << " times, Total time: " << total_time
              << "ms, Average time: " << total_time / M << "ms" << std::endl;

    return 0;
}