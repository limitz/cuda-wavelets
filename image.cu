#include <image.h>

Image::Defaults Image::Default;

// Lab color space
__constant__ __device__ float Lab_M[9];
__constant__ __device__ float Lab_Mi[9];
__constant__ __device__ float3 Lab_W;

#define Lab_e 0.008856f
#define Lab_k 903.3f
#define Lab_v 0.0031308f
#define Lab_vi 0.04045f

static void setup_cielab(cudaStream_t stream)
{
	static bool isInitialized = false;
	if (isInitialized) return;
	isInitialized = true;

	int rc;
        float pW[3] = { 0.95047f, 1.0f, 1.08883f };
        float pM[9] = {
                 0.4124f, 0.3576f, 0.1805f,
                 0.2126f, 0.7152f, 0.0722f,
                 0.0193f, 0.1192f, 0.9504f,
        };
        float pMi[9] = {
                 3.2406f,-1.5372f,-0.4986f,
                -0.9689f, 1.8758f, 0.0415f,
                 0.0557f, -0.2040, 1.0571f,
        };
        rc = cudaMemcpyToSymbolAsync(Lab_M, &pM, 9*sizeof(float), 0, cudaMemcpyHostToDevice,stream);
        if (cudaSuccess != rc) throw "Unable to copy cielab chromacity matrix";

        rc = cudaMemcpyToSymbolAsync(Lab_Mi, &pMi, 9*sizeof(float), 0, cudaMemcpyHostToDevice,stream);
        if (cudaSuccess != rc) throw "Unable to copy cielab inverted chromacity matrix";

        rc = cudaMemcpyToSymbolAsync(Lab_W, &pW, sizeof(float3), 0, cudaMemcpyHostToDevice, stream);
        if (cudaSuccess != rc) throw "Unable to copy cielab reference white"; 
}

__global__
void f_srgb_to_device(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        auto s = View<float4>(in, pitch_in, x, y);
	float4 d = make_float4(
			s.x < 0.04045f ? s.x/12.92f : pow((s.x + 0.055f)/1.055f, 2.4f),
			s.y < 0.04045f ? s.y/12.92f : pow((s.y + 0.055f)/1.055f, 2.4f),
			s.z < 0.04045f ? s.z/12.92f : pow((s.z + 0.055f)/1.055f, 2.4f),
			s.w);

	View<float4>(out, pitch_out, x, y) = d;
}
__global__
void f_device_to_srgb(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        auto s = View<float4>(in, pitch_in, x, y);
	float4 d = make_float4(
			s.x < 0.0031308f ? s.x*12.92f : pow(1.055f * s.x, 1.0f/2.4f)-0.055,
			s.y < 0.0031308f ? s.y*12.92f : pow(1.055f * s.y, 1.0f/2.4f)-0.055,
			s.z < 0.0031308f ? s.z*12.92f : pow(1.055f * s.z, 1.0f/2.4f)-0.055,
			s.w);

	View<float4>(out, pitch_out, x, y) = d;
}

__global__
void f_device_to_grayscale(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        auto s = View<float4>(in, pitch_in, x, y);
	float l = 0.2126f * s.x + 0.7152f * s.y + 0.0722f * s.z;
	View<float4>(out, pitch_out, x, y) = make_float4(l, l, l, s.w);
}

__global__
void f_srgb_to_grayscale(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        auto s = View<float4>(in, pitch_in, x, y);
	float4 d = make_float4(
			s.x < 0.04045f ? s.x/12.92f : pow((s.x + 0.055f)/1.055f, 2.4f),
			s.y < 0.04045f ? s.y/12.92f : pow((s.y + 0.055f)/1.055f, 2.4f),
			s.z < 0.04045f ? s.z/12.92f : pow((s.z + 0.055f)/1.055f, 2.4f),
			s.w);
	float l = 0.2126f * d.x + 0.7152f * d.y + 0.0722f * d.z;

	View<float4>(out, pitch_out, x, y) = make_float4(l,l,l,s.w);
}

__global__
void f_grayscale_to_srgb(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        auto s = View<float4>(in, pitch_in, x, y);
	float d = s.x < 0.0031308f ? s.x*12.92f : pow(1.055f * s.x, 1.0f/2.4f)-0.055;
	View<float4>(out, pitch_out, x, y) = make_float4(d,d,d, s.w);
}
__global__
void f_srgb_to_cielab(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        auto p = View<float4>(in, pitch_in, x, y);
        auto RGB = make_float3(p.x, p.y, p.z);

        float3 rgb = make_float3(
                RGB.x <= Lab_vi ? RGB.x / 12.92 : pow((RGB.x + 0.055)/1.055, 2.4),
                RGB.y <= Lab_vi ? RGB.y / 12.92 : pow((RGB.y + 0.055)/1.055, 2.4),
                RGB.z <= Lab_vi ? RGB.z / 12.92 : pow((RGB.z + 0.055)/1.055, 2.4)
        );

        float3 xyz = make_float3(
                Lab_M[0] * rgb.x + Lab_M[1] * rgb.y + Lab_M[2] * rgb.z,
                Lab_M[3] * rgb.x + Lab_M[4] * rgb.y + Lab_M[5] * rgb.z,
                Lab_M[6] * rgb.x + Lab_M[7] * rgb.y + Lab_M[8] * rgb.z
        );
        float3 r = make_float3(
                xyz.x / Lab_W.x,
                xyz.y / Lab_W.y,
                xyz.z / Lab_W.z
        );
        float3 f = make_float3(
                r.x > Lab_e ? pow(r.x, 1.0f/3.0f) : (Lab_k * r.x + 16.0f) / 116.0f,
                r.y > Lab_e ? pow(r.y, 1.0f/3.0f) : (Lab_k * r.y + 16.0f) / 116.0f,
                r.z > Lab_e ? pow(r.z, 1.0f/3.0f) : (Lab_k * r.z + 16.0f) / 116.0f
        );

        float4 Lab = make_float4(
                        116.0f * f.y - 16.0f,
                        500.0f * (f.x - f.y),
                        200.0f * (f.y - f.z),
			p.w);

        View<float4>(out, pitch_out, x, y) = Lab;
}

__global__
void f_cielab_to_srgb(float4* out, size_t pitch_out, float4* in, size_t pitch_in, size_t width, size_t height)
{
        int x = (blockIdx.x * blockDim.x + threadIdx.x);
        int y = (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) return;

        float4 Lab = View<float4>(in, pitch_in, x, y);
        float3 f = make_float3(
                (Lab.x+16.0f)/116.0f + Lab.y/500.0f,
                (Lab.x+16.0f)/116.0f,
                (Lab.x+16.0f)/116.0f - Lab.z/200.0f
        );
        float3 f3 = make_float3(
                f.x * f.x * f.x,
                f.y * f.y * f.y,
                f.z * f.z * f.z
        );
        float3 r = make_float3(
                f3.x > Lab_e ? f3.x : (116.0f * f.x - 16.0f)/Lab_k,
                f3.y > Lab_e ? f3.y : (116.0f * f.y - 16.0f)/Lab_k,
                f3.z > Lab_e ? f3.z : (116.0f * f.z - 16.0f)/Lab_k
                //Lab.x > Lab_k*Lab_e ? f3.y : Lab.x/Lab_k
        );
        float3 xyz = make_float3(
                r.x * Lab_W.x,
                r.y * Lab_W.y,
                r.z * Lab_W.z
        );
        float3 rgb = make_float3(
                Lab_Mi[0] * xyz.x + Lab_Mi[1] * xyz.y + Lab_Mi[2] * xyz.z,
                Lab_Mi[3] * xyz.x + Lab_Mi[4] * xyz.y + Lab_Mi[5] * xyz.z,
                Lab_Mi[6] * xyz.x + Lab_Mi[7] * xyz.y + Lab_Mi[8] * xyz.z
        );
        float3 RGB = make_float3(
                rgb.x <= Lab_v ? 12.92f * rgb.x : 1.055f * pow(rgb.x, 1.0f/2.4f) - 0.055,
                rgb.y <= Lab_v ? 12.92f * rgb.y : 1.055f * pow(rgb.y, 1.0f/2.4f) - 0.055,
                rgb.z <= Lab_v ? 12.92f * rgb.z : 1.055f * pow(rgb.z, 1.0f/2.4f) - 0.055
        );
        View<float4>(out, pitch_out, x, y) = 
		make_float4(
			clamp(RGB.x, 0.0f, 1.0f), 
			clamp(RGB.y, 0.0f, 1.0f),
			clamp(RGB.z, 0.0f, 1.0f),
			Lab.w
		);
}


static const char* strrstr(const char* c, const char* find)
{
	if (!c || !find) return nullptr;
	if (strlen(find) == 0) return c + strlen(c);
	if (strlen(c) < strlen(find)) return nullptr;

	for (int i=strlen(c)-strlen(find); i >= 0; i--)
	{
		if (!memcmp(c + i, find, strlen(find))) return c + i;
	}
	return nullptr;
}

Image::~Image()
{
	if (_filename) free(_filename);
	if (mem.host.data) cudaFreeHost(mem.host.data);
	if (mem.device.data) cudaFree(mem.device.data);
}

void Image::toHost(cudaStream_t stream)
{
	int rc = cudaMemcpy2DAsync(
			mem.host.data, mem.host.pitch, 
			mem.device.data, mem.device.pitch, 
			width * sizeof(float4), 
			height,
			cudaMemcpyDeviceToHost, 
			stream); 
	if (cudaSuccess != rc) throw "Unable to copy from device to host";
}
void Image::toDevice(cudaStream_t stream)
{
	int rc = cudaMemcpy2DAsync(
			mem.device.data, mem.device.pitch, 
			mem.host.data, mem.host.pitch, 
			width * sizeof(float4), 
			height,
			cudaMemcpyHostToDevice, 
			stream); 
	if (cudaSuccess != rc) throw "Unable to copy from host to device";
}

#define FROM_TO(a, b) ((int)(a)<<8|(int)(b))

void Image::convert(ColorSpace cs, cudaStream_t stream)
{
	if (cs == ColorSpace::Default) cs = Image::Default.colorSpace;
	setup_cielab(stream);
	
	dim3 blockSize = { 16, 16 };
	dim3 gridSize = {
		((int)width  + blockSize.x - 1) / blockSize.x,
		((int)height + blockSize.y - 1) / blockSize.y
	};

	switch (FROM_TO(colorSpace, cs))
	{
		case FROM_TO(ColorSpace::sRGB, ColorSpace::Device):
			f_srgb_to_device <<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;

		case FROM_TO(ColorSpace::Device, ColorSpace::sRGB):
			f_device_to_srgb <<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;
		case FROM_TO(ColorSpace::sRGB, ColorSpace::CIELab):
			f_srgb_to_cielab <<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;

		case FROM_TO(ColorSpace::CIELab, ColorSpace::sRGB):
			f_cielab_to_srgb <<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;
		
		case FROM_TO(ColorSpace::Grayscale, ColorSpace::sRGB):
			f_grayscale_to_srgb <<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;
		
		case FROM_TO(ColorSpace::sRGB, ColorSpace::Grayscale):
			f_srgb_to_grayscale<<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;
		
		case FROM_TO(ColorSpace::Device, ColorSpace::Grayscale):
			f_device_to_grayscale<<< gridSize, blockSize, 0, stream >>> (
				mem.device.data, mem.device.pitch,
				mem.device.data, mem.device.pitch,
				width, height);
			break;
		
		case FROM_TO(ColorSpace::Grayscale, ColorSpace::Device):
			//nop
			break;

		default: 
			throw "Conversion not implemented yet";
	}
	colorSpace = cs;
}

void Image::printInfo()
{
	printf("IMAGE %s\n", filename);
	printf("- SIZE:  %lu x %lu\n", width, height);
	printf("- PITCH: %lu (dev), %lu (host)\n", mem.device.pitch, mem.host.pitch);
	printf("\n");
	fflush(stdout);
}

void Image::alloc(size_t w, size_t h, size_t c)
{
	int rc;
	if (mem.host.data) cudaFreeHost(mem.host.data);
	if (mem.device.data) cudaFree(mem.device.data);

	width = w;
	height = h;
	channels = c;

	mem.host.pitch = width * sizeof(float4);
	rc = cudaMallocHost(&mem.host.data, mem.host.pitch * height);
	if (cudaSuccess != rc) throw "Unable to allocate host memory for image";

	rc = cudaMallocPitch(&mem.device.data, &mem.device.pitch, 
			width * sizeof(float4), height);
	if (cudaSuccess != rc) throw "Unable to allocate device memory for image";
}

void Image::loadPPM()
{
	int rc;

	FILE* f = fopen(filename, "rb");
	if (!f) throw "Unable to open file";

	size_t range;
	rc = fscanf(f, "P6 %lu %lu %lu \n", &width, &height, &range);
	if (rc <= 0) throw "Unable to read PPM header";

	int bpp;
	channels = 3;
	if (range < 256) bpp = 8;
	else bpp = 16;

	mem.host.pitch = width * sizeof(float4);
	size_t pitch = width * (bpp >> 3) * channels;
	rc = cudaMallocHost(&mem.host.data, width * sizeof(float4) * height);
	if (cudaSuccess != rc) throw "Unable to allocate host memory for image";

	rc = cudaMallocPitch(&mem.device.data, &mem.device.pitch, width * sizeof(float4), height);
	if (cudaSuccess != rc) throw "Unable to allocate device memory for image";

	rc = fread(mem.host.data, 1, pitch * height, f);
	if (rc <= 0) throw "Unable to read image data from PPM";

	for (int i= width * height - 1; i>= 0; i--)
	{
		if (bpp == 8)
		{
			uchar3 p = ((uchar3*)mem.host.data)[i];
			mem.host.data[i] = make_float4(p.x, p.y, p.z, range) / (float)range;
		}
		else
		{
			ushort3 p = ((ushort3*)mem.host.data)[i];
			mem.host.data[i] = make_float4(p.x, p.y, p.z, range) / (float)range;
		}
	}
	fclose(f);
	toDevice();
}

void Image::loadPGM()
{
	throw "Not implemented yet";
}

void Image::loadJPG()
{
	throw "Not implemented yet";
}

void Image::load(const char* path)
{
	filename = _filename = strdup(path);

	const char* extension = strrstr(filename, ".");

	if (!strcmp(extension, ".jpeg")) loadJPG();
	if (!strcmp(extension, ".jpg"))  loadJPG();
	if (!strcmp(extension, ".ppm"))  loadPPM();
	if (!strcmp(extension, ".pgm"))  loadPGM();
}

float Image::psnr(const Image* ref)
{
	float mse = 0;
	for (size_t x=2; x<width-2; x++)
	{
		for (size_t y=2; y<height-2; y++)
		{
			for (size_t c=0; c<channels; c++)
			{
				void* p = ((uint8_t*)mem.host.data
						+ y * mem.host.pitch
						+ (x * sizeof(float4)));

				void* q = ((uint8_t*)ref->mem.host.data
						+ y * ref->mem.host.pitch
						+ (x * sizeof(float4)));
				float pv = ((float*)p)[c];
				float qv = ((float*)q)[c];

				mse += (pv - qv) * (pv - qv);
			}
		}
	}
	mse /= width * height * channels;
	return 20 * log10(1.0) - 10 * log10(mse);
}

JpegCodec::JpegCodec()
{
	_width = 0;
	_height = 0;
	_channels = 0;
	_buffer = nullptr;
	_scanlines = nullptr;
	
	_dinfo.err = jpeg_std_error(&_djerr);
	_cinfo.err = jpeg_std_error(&_cjerr);
}
	
JpegCodec::~JpegCodec()
{
	free(_buffer);
	free(_scanlines);
}

void JpegCodec::prepare(int width, int height, int channels, int quality)
{
	if (channels != 3) throw "Not implemented channels != 3";

	_width = width;
	_height = height;
	_channels = channels;

	_buffer = (uint8_t*) malloc(_width * _height * _channels);
	if (!_buffer) throw "Unable to allocate intermediate buffer";

	_scanlines = (JSAMPARRAY) malloc( sizeof(JSAMPROW) * height);
	if (!_scanlines)
	{
		free(_buffer);
		throw "Unable to allocate scanlines structure";
	}

	for (size_t i=0; i<_height; i++)
	{
		_scanlines[i] = (JSAMPROW) (_buffer + i * _width * _channels);
	}

	jpeg_create_decompress(&_dinfo);
	jpeg_create_compress(&_cinfo);
	
	_cinfo.image_width = _width;
	_cinfo.image_height = height;
	_cinfo.input_components = 3;
	_cinfo.in_color_space = JCS_RGB; 
	jpeg_set_defaults(&_cinfo);
	jpeg_set_quality(&_cinfo, quality, 1);
}

void JpegCodec::unprepare()
{
	jpeg_destroy_decompress(&_dinfo);
	jpeg_destroy_compress(&_cinfo);
}

void JpegCodec::encodeCPU(void* dst, size_t *size)
{
	//cudaMemcpyAsync(_buffer, src, _width * _height * _channels, cudaMemcpyDeviceToHost, stream);
	//cudaStreamSynchronize(stream);
	
	jpeg_mem_dest(&_cinfo, (uint8_t**)&dst, size);
	jpeg_start_compress(&_cinfo, 1);
	while (_cinfo.next_scanline < _cinfo.image_height)
	{
		jpeg_write_scanlines(&_cinfo, _scanlines + _cinfo.next_scanline, _cinfo.image_height - _cinfo.next_scanline);
	}
	jpeg_finish_compress(&_cinfo);
}

void JpegCodec::decodeToDeviceMemoryCPU(void* dst, const void* src, size_t size, cudaStream_t stream)
{
	jpeg_mem_src(&_dinfo, (uint8_t*)src, size);
	jpeg_read_header(&_dinfo, 1);
	jpeg_calc_output_dimensions(&_dinfo);

	if (_dinfo.output_width != _width 
	||  _dinfo.output_height != _height
	||  _dinfo.output_components != (int) _channels)
	{
		jpeg_abort_decompress(&_dinfo);
		throw "Invalid image format";
	}
	jpeg_start_decompress(&_dinfo);
	while (_dinfo.output_scanline < _dinfo.output_height)
	{
		jpeg_read_scanlines(&_dinfo, _scanlines + _dinfo.output_scanline,_dinfo.output_height - _dinfo.output_scanline);
	}
	jpeg_finish_decompress(&_dinfo);

	cudaMemcpyAsync(dst, _buffer, _width * _height * _channels, cudaMemcpyHostToDevice, stream);
}

#if USE_NVJPEG
void JpegCodec::decodeToDeviceMemoryGPU(void* dst, const void* src, size_t size, cudaStream_t stream)
{
	int rc;
	
	nvjpegHandle_t handle;
	rc = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, 0, &handle);
	if (cudaSuccess != rc) throw "Unable to create nvjpeg handle";

	int channels;
	int widths[NVJPEG_MAX_COMPONENT];
	int heights[NVJPEG_MAX_COMPONENT];
	nvjpegChromaSubsampling_t subsampling;
	nvjpegJpegState_t state;
	nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI;
	nvjpegJpegStateCreate(handle, &state);
	nvjpegGetImageInfo(handle, (uint8_t*) src, size, &channels, &subsampling, widths, heights);

	if (widths[0] != (int)_width
	||  heights[0] != (int)_height)
	{
		nvjpegJpegStateDestroy(state);
		nvjpegDestroy(handle);
		throw "Invalid image format";
	}

	nvjpegImage_t output;
	output.channel[0] = (uint8_t*) dst;
	output.pitch[0] = widths[0] * _channels;

	nvjpegDecode(handle, state, (uint8_t*)src, size, fmt, &output, stream);
	nvjpegJpegStateDestroy(state);
	nvjpegDestroy(handle);

}
#endif
