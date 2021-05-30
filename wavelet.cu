#include <wavelet.h>
#include <daubechies.h>

Wavelet::Wavelet(const char* name)
{
	if (!strcmp(name, "db1" )) { const float mother[] = { DB1  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db2" )) { const float mother[] = { DB2  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db3" )) { const float mother[] = { DB3  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db4" )) { const float mother[] = { DB4  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db5" )) { const float mother[] = { DB5  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db6" )) { const float mother[] = { DB6  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db7" )) { const float mother[] = { DB7  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db8" )) { const float mother[] = { DB8  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db9" )) { const float mother[] = { DB9  }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db10")) { const float mother[] = { DB10 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db11")) { const float mother[] = { DB11 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db12")) { const float mother[] = { DB12 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db13")) { const float mother[] = { DB13 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db14")) { const float mother[] = { DB14 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db15")) { const float mother[] = { DB15 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db16")) { const float mother[] = { DB16 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db17")) { const float mother[] = { DB17 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db18")) { const float mother[] = { DB18 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db19")) { const float mother[] = { DB19 }; init(mother, sizeof(mother)/sizeof(float)); }
	if (!strcmp(name, "db20")) { const float mother[] = { DB20 }; init(mother, sizeof(mother)/sizeof(float)); }
}

Wavelet::~Wavelet()
{
	if (decompose.H.device) cudaFree(decompose.H.device);
	if (decompose.H.host) cudaFreeHost(decompose.H.host);
}

void Wavelet::init(const float* mother, size_t len)
{
	length = len;
	size_t pitch = len * sizeof(float);

	int rc;
	rc = cudaMallocHost(&decompose.H.host, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on host";
	
	rc = cudaMallocHost(&decompose.G.host, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on host";
	
	rc = cudaMallocHost(&reconstruct.H.host, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on host";
	
	rc = cudaMallocHost(&reconstruct.G.host, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on host";
	
	rc = cudaMalloc(&decompose.H.device, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on device";
	
	rc = cudaMalloc(&decompose.G.device, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on device";
	
	rc = cudaMalloc(&reconstruct.H.device, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on device";
	
	rc = cudaMalloc(&reconstruct.G.device, pitch);
	if (cudaSuccess != rc) throw "Unable to allocate wavelet on device";

	for (size_t i = 0; i < len; i++)
	{
		decompose.H.host[i] = mother[i];
		decompose.G.host[i] = mother[len - i - 1] * ((i % 2) ? -1 : 1);
		reconstruct.H.host[i] = mother[len - i - 1];
		reconstruct.G.host[i] = mother[i] * ((i % 2) ? -1 : 1);
	}

	rc = cudaMemcpy(decompose.H.device, decompose.H.host, pitch, cudaMemcpyHostToDevice);
	if (cudaSuccess != rc) throw "Unable to copy wavelet to device";
	rc = cudaMemcpy(decompose.G.device, decompose.G.host, pitch, cudaMemcpyHostToDevice);
	if (cudaSuccess != rc) throw "Unable to copy wavelet to device";
	rc = cudaMemcpy(reconstruct.H.device, reconstruct.H.host, pitch, cudaMemcpyHostToDevice);
	if (cudaSuccess != rc) throw "Unable to copy wavelet to device";
	rc = cudaMemcpy(reconstruct.G.device, reconstruct.G.host, pitch, cudaMemcpyHostToDevice);
	if (cudaSuccess != rc) throw "Unable to copy wavelet to device";
	cudaDeviceSynchronize();
}

__global__ void f_decompose_h(float4* out, const size_t pitch_out, float4* in, const size_t pitch_in, const size_t width, const size_t height, const float* H, const float* G, const size_t len)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

	auto s = View2DSym<float4>(in, pitch_in, x-(len-1)/2, y, width, height);
	float4 A = make_float4(0);
	float4 D = make_float4(0);

	#pragma unroll
	for (size_t i = 0; i < len; i++) 
	{
		auto v = s(i,0);
		A += v * H[i];
		D += v * G[i];
	}
	View<float4>(out, pitch_out, x>>1, y) = A * rsqrtf(2);
	View<float4>(out, pitch_out, (width + x)>>1, y) = D * rsqrtf(2);
}

__global__ void f_decompose_v(float4* out, const size_t pitch_out, float4* in, const size_t pitch_in, const size_t width, const size_t height, const float* H, const float* G, const size_t len)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        if (x >= width || y >= height) return;

	auto s = View2DSym<float4>(in, pitch_in, x, y, width, height);
	float4 A = make_float4(0);
	float4 D = make_float4(0);
	
	#pragma unroll
	for (size_t i = 0; i < len; i++) 
	{
		auto v = s(0,i);
		A += v * H[i];
		D += v * G[i];
	}
	View<float4>(out, pitch_out, x, y>>1) = A * rsqrtf(2);
	View<float4>(out, pitch_out, x, (y+height)>>1) = D * rsqrtf(2);
}

__global__ void f_reconstruct_h(float4* out, const size_t pitch_out, float4* in, const size_t pitch_in, const size_t width, const size_t height, const float* H, const float* G, const size_t len)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x)*2;
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

	auto s = View2DSym<float4>(in, pitch_in, (x>>1) - ((len-1)/2), y, width, height);
	float4 A = make_float4(0);
	float4 D = make_float4(0);
	#pragma unroll
	for (size_t i = 0; i < len; i++) 
	{
		auto v = s((i>>1) + (i%2) * (width>>1),0);
		A += v * H[i];
		D += v * G[i];
	}

	View<float4>(out, pitch_out, x, y) = A * sqrtf(2);
	View<float4>(out, pitch_out, x+1, y) = D * sqrtf(2);
}

__global__ void f_reconstruct_v(float4* out, const size_t pitch_out, float4* in, const size_t pitch_in, const size_t width, const size_t height, const float* H, const float* G, const size_t len)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y)*2;
	if (x >= width || y >= height) return;

	auto s = View2DSym<float4>(in, pitch_in, x, (y>>1) - ((len-1)/2), width, height);
	float4 A = make_float4(0);
	float4 D = make_float4(0);
	#pragma unroll
	for (size_t i = 0; i < len; i++) 
	{
		auto v = s(0, (i>>1) + (i%2) * (height>>1));
		A += v * H[i];
		D += v * G[i];
	}

	View<float4>(out, pitch_out, x, y) = A * sqrtf(2);
	View<float4>(out, pitch_out, x, y+1) = D * sqrtf(2);
}

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
	if (!intermediate1 || intermediate1->width != destination->width || intermediate1->height != destination->height)
	{
		if (intermediate1) delete intermediate1;
		intermediate1 = new Image(destination->colorSpace, destination->width, destination->height, destination->channels);
	}

	dim3 blockSize = { 16, 16 };
	int w = (int)source->width;
	int h = (int)source->height;

	size_t i = 0;
	//for (size_t i = 0; i < levels; i++)
	{
		Image* src = i ? destination : source;
		Image* sub = intermediate1;
		Image* dst = destination;

		dim3 gridSizeH = {
			1+((w>>1) + blockSize.x-1) / blockSize.x, 
			1+(h + blockSize.y - 1) / blockSize.y
		};
		dim3 gridSizeV = {
			1+(w + blockSize.x - 1) / blockSize.x, 
			1+((h>>1) + blockSize.y-1) / blockSize.y
		};
		
		f_decompose_h <<< gridSizeH, blockSize, 0, stream >>> (
			sub->mem.device.data, sub->mem.device.pitch,
			src->mem.device.data, src->mem.device.pitch,
			w, h,
			wavelet->decompose.H.device, 
			wavelet->decompose.G.device, 
			wavelet->length);
		
		f_decompose_v <<< gridSizeV, blockSize, 0, stream >>> (
			dst->mem.device.data, dst->mem.device.pitch,
			sub->mem.device.data, sub->mem.device.pitch,
			w, h,
			wavelet->decompose.H.device, 
			wavelet->decompose.G.device, 
			wavelet->length);
	
		f_reconstruct_v <<< gridSizeV, blockSize, 0, stream >>> (
			sub->mem.device.data, sub->mem.device.pitch,
			dst->mem.device.data, dst->mem.device.pitch,
			w, h,
			wavelet->reconstruct.H.device, 
			wavelet->reconstruct.G.device,
			wavelet->length);
		
		f_reconstruct_h <<< gridSizeH, blockSize, 0, stream >>> (
			dst->mem.device.data, dst->mem.device.pitch,
			sub->mem.device.data, sub->mem.device.pitch,
			w, h,
			wavelet->reconstruct.H.device, 
			wavelet->reconstruct.G.device,
			wavelet->length);
		w >>= 1;
		h >>= 1;
	}
}
