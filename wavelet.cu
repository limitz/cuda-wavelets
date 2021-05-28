#include <wavelet.h>

__global__
void f_rearrange(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height, size_t levels)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;
        auto s = View<float4>(in, pitch_in, x, y);
	int scale = __ffs(y|x);

	int kx = (1 & (x>>(scale-1))) * (width>>(scale)) + (x >> (scale));
	int ky = (1 & (y>>(scale-1))) * (height>>(scale)) + (y >> (scale));

	View<float4>(out, pitch_out, kx, ky) = s;
}

void WaveletTransform2D::run(cudaStream_t stream)
{
	dim3 blockSize = { 16, 16 };
	int w = (int)source->width;
	int h = (int)source->height;


	dim3 gridSize = {
		(w + blockSize.x - 1) / blockSize.x, 
		(h + blockSize.y - 1) / blockSize.y
	};
	
	f_rearrange <<< gridSize, blockSize, 0, stream >>> (
			destination->mem.device.data, destination->mem.device.pitch,
			source->mem.device.data, source->mem.device.pitch,
			w, h, 10);
}
