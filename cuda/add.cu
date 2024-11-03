#include <stdio.h>

__global__ void add(int *a,  int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] +  b[blockIdx.x];
}


int main()  {
    int dim = 3;
    size_t size =  dim * sizeof(int);
    int *x = (int *)malloc(size);
    int *y = (int *)malloc(size);

    int *d_x, *d_y; 
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    
    x[0] = 0;
    x[1] = 1;
    x[2] = 2;

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice); 

    add<<<dim, 1>>>(d_x, d_x, d_y);
    
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    printf("%d  %d  %d\n", y[0], y[1], y[2]);

    free(x);
    free(y);

    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}