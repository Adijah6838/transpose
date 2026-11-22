// transpose.cu - Naive vs Tiled Shared-Memory Matrix Transpose
// Compile: nvcc -O3 -arch=sm_80 transpose.cu -o transpose
// Run: ./transpose [N]   e.g. ./transpose 8192
// Nsight: ncu --set full -o naive ./transpose 4096 --naive
//         ncu --set full -o tiled ./transpose 4096 --tiled
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define TILE 32
#define CHECK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1;}}while(0)
// Naive: read row (coalesced), write column (strided / uncoalesced)
__global__ void transpose_naive(const float* in, float* out, int N){
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<N && y<N) out[x*N+y]=in[y*N+x];
}
// Tiled: coalesced read -> shared (padded +1 to avoid bank conflicts) -> coalesced write
__global__ void transpose_tiled(const float* in, float* out, int N){
  __shared__ float tile[TILE][TILE+1];
  int x=blockIdx.x*TILE+threadIdx.x;
  int y=blockIdx.y*TILE+threadIdx.y;
  if(x<N && y<N) tile[threadIdx.y][threadIdx.x]=in[y*N+x];
  __syncthreads();
  int ox=blockIdx.y*TILE+threadIdx.x;
  int oy=blockIdx.x*TILE+threadIdx.y;
  if(ox<N && oy<N) out[oy*N+ox]=tile[threadIdx.x][threadIdx.y];
}
void cpu_transpose(const float* in, float* out, int N){
  for(int y=0;y<N;y++) for(int x=0;x<N;x++) out[x*N+y]=in[y*N+x];
}
int main(int argc,char** argv){
  int N=argc>1?atoi(argv[1]):8192;
  bool onlyNaive=argc>2?std::string(argv[2])=="--naive":false;
  bool onlyTiled=argc>2?std::string(argv[2])=="--tiled":false;
  printf("N=%d (%dx%d) size=%.1f MB\n",N,N,N,(float)N*N*4/1e6);
  std::vector<float> h_in((size_t)N*N), h_cpu((size_t)N*N), h_out((size_t)N*N);
  srand(42); for(size_t i=0;i<h_in.size();i++) h_in[i]=(float)(rand()%1000)/10.0f;
  float *d_in,*d_out; CHECK(cudaMalloc(&d_in,(size_t)N*N*4)); CHECK(cudaMalloc(&d_out,(size_t)N*N*4));
  CHECK(cudaMemcpy(d_in,h_in.data(),(size_t)N*N*4,cudaMemcpyHostToDevice));
  dim3 block(TILE,TILE); dim3 grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);
  auto bench=[&](auto kern,const char* name){
    kern<<<grid,block>>>(d_in,d_out,N); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for(int r=0;r<20;r++) kern<<<grid,block>>>(d_in,d_out,N); cudaEventRecord(b); CHECK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,a,b); ms/=20;
    CHECK(cudaMemcpy(h_out.data(),d_out,(size_t)N*N*4,cudaMemcpyDeviceToHost));
    double gbs=(2.0*N*N*4)/(ms*1e6);
    printf("%-8s %7.3f ms  %7.2f GB/s\n",name,ms,gbs);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return gbs;
  };
  double gNaive=0,gTiled=0;
  if(!onlyTiled) gNaive=bench(transpose_naive,"naive");
  if(!onlyNaive) gTiled=bench(transpose_tiled,"tiled");
  // verify tiled vs naive by re-running both and comparing
  transpose_naive<<<grid,block>>>(d_in,d_out,N); CHECK(cudaDeviceSynchronize());
  std::vector<float> h_naive((size_t)N*N); CHECK(cudaMemcpy(h_naive.data(),d_out,(size_t)N*N*4,cudaMemcpyDeviceToHost));
  transpose_tiled<<<grid,block>>>(d_in,d_out,N); CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h_out.data(),d_out,(size_t)N*N*4,cudaMemcpyDeviceToHost));
  double maxErr=0; for(size_t i=0;i<h_out.size();i++) maxErr=fmax(maxErr,fabs((double)h_naive[i]-h_out[i]));
  printf("naive vs tiled maxErr=%.2e %s\n",maxErr,maxErr<1e-5?"PASS":"FAIL");
  if(gNaive>0 && gTiled>0) printf("Speedup: %.2fx (tiled %.1f GB/s vs naive %.1f GB/s)\n",gTiled/gNaive,gTiled,gNaive);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaDeviceReset());
  return maxErr<1e-5?0:1;
}
void cpu_transpose(const float* in, float* out, int N){
  for(int y=0;y<N;y++) for(int x=0;x<N;x++) out[x*N+y]=in[y*N+x];
}
int main(int argc,char** argv){
  int N=argc>1?atoi(argv[1]):8192;
  bool onlyNaive=argc>2?std::string(argv[2])=="--naive":false;
  bool onlyTiled=argc>2?std::string(argv[2])=="--tiled":false;
  printf("N=%d (%dx%d) size=%.1f MB\n",N,N,N,(float)N*N*4/1e6);
  std::vector<float> h_in((size_t)N*N), h_cpu((size_t)N*N), h_out((size_t)N*N);
  srand(42); for(size_t i=0;i<h_in.size();i++) h_in[i]=(float)(rand()%1000)/10.0f;
  float *d_in,*d_out; CHECK(cudaMalloc(&d_in,(size_t)N*N*4)); CHECK(cudaMalloc(&d_out,(size_t)N*N*4));
  CHECK(cudaMemcpy(d_in,h_in.data(),(size_t)N*N*4,cudaMemcpyHostToDevice));
  dim3 block(TILE,TILE); dim3 grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);
  auto bench=[&](auto kern,const char* name){
    kern<<<grid,block>>>(d_in,d_out,N); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for(int r=0;r<20;r++) kern<<<grid,block>>>(d_in,d_out,N); cudaEventRecord(b); CHECK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,a,b); ms/=20;
    CHECK(cudaMemcpy(h_out.data(),d_out,(size_t)N*N*4,cudaMemcpyDeviceToHost));
    double gbs=(2.0*N*N*4)/(ms*1e6);
    printf("%-8s %7.3f ms  %7.2f GB/s\n",name,ms,gbs);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return gbs;
  };
  double gNaive=0,gTiled=0;
  if(!onlyTiled) gNaive=bench(transpose_naive,"naive");
  if(!onlyNaive) gTiled=bench(transpose_tiled,"tiled");
  // verify tiled vs naive by re-running both and comparing
  transpose_naive<<<grid,block>>>(d_in,d_out,N); CHECK(cudaDeviceSynchronize());
  std::vector<float> h_naive((size_t)N*N); CHECK(cudaMemcpy(h_naive.data(),d_out,(size_t)N*N*4,cudaMemcpyDeviceToHost));
  transpose_tiled<<<grid,block>>>(d_in,d_out,N); CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h_out.data(),d_out,(size_t)N*N*4,cudaMemcpyDeviceToHost));
  double maxErr=0; for(size_t i=0;i<h_out.size();i++) maxErr=fmax(maxErr,fabs((double)h_naive[i]-h_out[i]));
  printf("naive vs tiled maxErr=%.2e %s\n",maxErr,maxErr<1e-5?"PASS":"FAIL");
  if(gNaive>0 && gTiled>0) printf("Speedup: %.2fx (tiled %.1f GB/s vs naive %.1f GB/s)\n",gTiled/gNaive,gTiled,gNaive);
  CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaDeviceReset());
  return maxErr<1e-5?0:1;
}
