#include <gpu.h>
#include <image.h>
#include <display.h>

__global__
void f_add_uchar1_to_float4(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, float4 components, uint8_t range = 255)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	float px = View<uint8_t>(in, pitch_in, x, y) / (float)range;
	View<float4>(out, pitch_out, x, y) += components * px;
}
__global__
void f_set_uchar1_to_float4(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, float4 components, uint8_t range = 255)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	float px = View<uint8_t>(in, pitch_in, x, y) / (float)range;
	View<float4>(out, pitch_out, x, y) = components * px;
}

__global__
void f_set_uchar3_to_float4(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, uint8_t range = 255)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	uchar3 p = View<uchar3>(in, pitch_in, x, y);
	View<float4>(out, pitch_out, x, y) = make_float4(p.x, p.y, p.z, 0.0) / (float)range;
}

__global__
void f_set_float4_to_uchar1(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, float4 components, uint8_t range = 255)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	
	float4 p = View<float4>(in, pitch_in, x, y) * components;
	float px = (p.x + p.y + p.z + p.w) * range;
	View<uint8_t>(out, pitch_out, x, y) = (uint8_t) clamp((int)px, 0, 255 );
}

__global__
void f_set_float4_to_uchar3(float4* out, size_t pitch_out, void* in, size_t pitch_in, size_t width, size_t height, uint8_t range = 255)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	float4 p = View<float4>(in, pitch_in, x, y) * range;
	View<uchar3>(out, pitch_out, x, y) = clamp(make_uchar3(p.x, p.y, p.z), 0, range);
}

static int smToCores(int major, int minor)
{
	switch ((major << 4) | minor)
	{
		case (9999 << 4 | 9999):
			return 1;
		case 0x30:
		case 0x32:
		case 0x35:
		case 0x37:
			return 192;
		case 0x50:
		case 0x52:
		case 0x53:
			return 128;
		case 0x60:
			return 64;
		case 0x61:
		case 0x62:
			return 128;
		case 0x70:
		case 0x72:
		case 0x75:
			return 64;
		case 0x80:
			return 64;
		case 0x86:
			return 128;
		default:
			return 0;
	};
}

void selectGPU()
{
	int rc;
	int maxId = -1;
	uint16_t maxScore = 0;
	int count = 0;
	cudaDeviceProp prop;

	rc = cudaGetDeviceCount(&count);
	if (cudaSuccess != rc) throw "cudaGetDeviceCount error";
	if (count == 0) throw "No suitable cuda device found";

	for (int id = 0; id < count; id++)
	{
		rc = cudaGetDeviceProperties(&prop, id);
		if (cudaSuccess != rc) throw "Unable to get device properties";
		if (prop.computeMode == cudaComputeModeProhibited) 
		{
			printf("GPU %d: PROHIBITED\n", id);
			continue;
		}
		int sm_per_multiproc = smToCores(prop.major, prop.minor);
		
		printf("GPU %d: \"%s\"\n", id, prop.name);
		printf(" - Compute capability: %d.%d\n", prop.major, prop.minor);
		printf(" - Multiprocessors:    %d\n", prop.multiProcessorCount);
		printf(" - SMs per processor:  %d\n", sm_per_multiproc);
		printf(" - Clock rate:         %d\n", prop.clockRate);

		uint64_t score =(uint64_t) prop.multiProcessorCount * sm_per_multiproc * prop.clockRate;
		if (score > maxScore) 
		{
			maxId = id;
			maxScore = score;
		}
	}

	if (maxId < 0) throw "All cuda devices prohibited";

	rc = cudaSetDevice(maxId);
	if (cudaSuccess != rc) throw "Unable to set cuda device";

	rc = cudaGetDeviceProperties(&prop, maxId);
	if (cudaSuccess != rc) throw "Unable to get device properties";

	printf("\nSelected GPU %d: \"%s\" with compute capability %d.%d\n\n", 
		maxId, prop.name, prop.major, prop.minor);
}
