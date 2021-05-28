#include <wavelet.h>

__global__
void f_rearrange_quad(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height, size_t level)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        if (x >= width || y >= height) return;
        auto s = View2DSym<float4>(in, pitch_in, x, y,width <<level, height << level);
	auto a = (s(0,0) + s(1,0) + s(0,1) + s(1,1))/4;
	auto h = (s(1,0)-a)/2 + 0.5f;
	auto v = (s(0,1)-a)/2 + 0.5f;
	auto d = (s(1,1)-a)/2 + 0.5f;

	x >>= 1;
	y >>= 1;
	int kx = x + (width >>1);
	int ky = y + (height>>1);

	View<float4>(out, pitch_out, x, y) = a;
        View<float4>(out, pitch_out, kx, y) = h;
        View<float4>(out, pitch_out, x, ky) = v;
        View<float4>(out, pitch_out, kx, ky) = d;
}

void WaveletTransform2D::run(cudaStream_t stream)
{
	dim3 blockSize = { 16, 16 };
	int w = (int)source->width;
	int h = (int)source->height;


	Image intermediate(ColorSpace::Default, destination->width, destination->height, destination->channels);
	Image* s = source;
	Image* d = destination;
	for (int i=2; i>=0; i--)
	{
		dim3 gridSize = {
			(w/2 + blockSize.x - 1) / blockSize.x, 
			(h/2 + blockSize.y - 1) / blockSize.y
		};
		f_rearrange_quad <<< gridSize, blockSize, 0, stream >>> (
			d->mem.device.data, d->mem.device.pitch,
			s->mem.device.data, s->mem.device.pitch,
			w, h, i);
		gridSize.x >>= 1;
		gridSize.y >>= 1;
		w >>= 1;
		h >>= 1;
	
		if (i)  cudaMemcpy2DAsync(
			intermediate.mem.device.data, intermediate.mem.device.pitch, 
			destination->mem.device.data, destination->mem.device.pitch,
			w * sizeof(float4),
			h,
			cudaMemcpyDeviceToDevice,
			stream);
		s = &intermediate;
	}
}
