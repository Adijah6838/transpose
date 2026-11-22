import random
def transpose_cpu(m,N):
    return [m[y*N+x] for x in range(N) for y in range(N)]
def transpose_tiled_mirror(m,N,TILE=32):
    out=[0]*(N*N)
    for by in range(0,N,TILE):
        for bx in range(0,N,TILE):
            tile=[[0]*(TILE+1) for _ in range(TILE)]
            for ty in range(TILE):
                for tx in range(TILE):
                    x,y=bx+tx,by+ty
                    if x<N and y<N: tile[ty][tx]=m[y*N+x]
            for ty in range(TILE):
                for tx in range(TILE):
                    ox,oy=by+tx,bx+ty
                    if ox<N and oy<N: out[oy*N+ox]=tile[tx][ty]
    return out
for N in [64,100,256]:
    random.seed(42); m=[random.uniform(-10,10) for _ in range(N*N)]
    a=transpose_cpu(m,N); b=transpose_tiled_mirror(m,N); err=max(abs(x-y) for x,y in zip(a,b))
    print(f'N={N} PASS={err<1e-9} maxErr={err:.2e}')
print('Transpose validated - tiled == naive')
