# Matrix Transpose

Naive vs tiled shared-memory transpose - coalescing demo.

## Coalescing
Naive: read row = coalesced, write column = strided (32x wasted bandwidth).
Tiled: 32x32 tile, coalesced read -> shared -> coalesced write.
Padding: tile[32][33] - +1 column kills 32-way bank conflicts on transpose read.

## Benchmark (expected, T4 Colab, N=8192, 256MB)
| Kernel | Time | GB/s |
| Naive | ~8.5ms | ~63 |
| Tiled | ~2.1ms | ~254 |
| Speedup | ~4x | |
Fill real numbers from Colab run.

## Nsight Compute
ncu --set full -o naive ./transpose 4096 --naive
ncu --set full -o tiled ./transpose 4096 --tiled
Compare: Memory Throughput, L2 Cache Hit Rate - tiled higher, DRAM lower.

## Build / Run
nvcc -O3 -arch=sm_80 transpose.cu -o transpose
./transpose 8192
