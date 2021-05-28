#pragma once
#include <cuda_runtime.h>
#include <operators.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <gpu.h>

#if USE_NVJPEG 
#include <nvjpeg.h>
#endif
#include <jpeglib.h>

enum ColorSpace
{
	Default,
	Device,
	Grayscale,
	sRGB,
	CIELab
};

class Image
{
public:
	size_t width = 0, height = 0, channels = 0;
	const char* filename = nullptr;
	ColorSpace colorSpace = ColorSpace::Device;
	cudaStream_t stream = NULL;

	static Image like(const Image& img)
	{
		Image result(img.colorSpace, img.width, img.height, img.channels);
		result.stream = img.stream;
		return result;
	}

	struct
	{
		struct
		{
			float4* data = nullptr;
			size_t pitch = 0;
		} host, device;
	} mem;

	static struct Defaults
	{
		ColorSpace colorSpace;
		size_t channels = 3;
		size_t width = 1920;
		size_t height = 1080;
		cudaStream_t stream = NULL;
	} Default;

	Image()
	{
		colorSpace = Default.colorSpace;
		stream = Default.stream;
		alloc(Default.width, Default.height, Default.channels);
	}
	Image(ColorSpace cs)
	{
		colorSpace = cs == ColorSpace::Default ? Default.colorSpace : cs;
		if (colorSpace == ColorSpace::Grayscale) channels = 1;
		else channels = 3;

		stream = Default.stream;
		alloc(Default.width, Default.height, Default.channels);
	}

	Image(size_t width, size_t height)
	{
		colorSpace = Default.colorSpace;
		stream = Default.stream;
		alloc(width, height, Default.channels);
	}

	Image(ColorSpace cs, size_t width, size_t height, size_t channels = 0)
	{
		colorSpace = cs == ColorSpace::Default ? Default.colorSpace : cs;
		stream = Default.stream;
		alloc(width, height, channels <= 0 ? Default.channels : channels);
	}

	Image(const char* path, ColorSpace cs = ColorSpace::Default)
	{
		colorSpace = cs == ColorSpace::Default ? Default.colorSpace : cs;
		stream = Default.stream;
		load(path);
	}

	~Image();

	void alloc(size_t width, size_t height, size_t channels);
	void synchronize(cudaStream_t s = NULL) { cudaStreamSynchronize(s ? s : stream); }

	void load(const char* filename = nullptr);
	void save(const char* filename = nullptr);

	void toDevice() { toDevice(stream); }
	void toDevice(cudaStream_t s);

	void toHost() { toHost(stream); }
	void toHost(cudaStream_t s);
	
	void convert(ColorSpace cs) { convert(cs, stream);}
	void convert(ColorSpace cs, cudaStream_t s);
	void printInfo();

	float psnr(const Image* reference);

private:
	void loadPPM();
	void loadPGM();
	void loadJPG();
	char* _filename = nullptr;

};

class JpegCodec
{
public:
	
	JpegCodec();
	~JpegCodec();

	void* buffer() const { return _buffer; }

	void prepare(int width, int height, int channels, int quality);
	void unprepare();
	void decodeToDeviceMemoryCPU(void* dst, const void* src, size_t size, cudaStream_t stream);
	void decodeToDeviceMemoryGPU(void* dst, const void* src, size_t size, cudaStream_t stream);
	void encodeToHostMemoryGPU(void* dst, const void* src, size_t *size, cudaStream_t stream);
	void encodeToHostMemoryCPU(void* dst, const void* src, size_t *size, cudaStream_t stream);
	void encodeCPU(void* dst, size_t *size);

private:
	struct jpeg_decompress_struct _dinfo;
	struct jpeg_compress_struct _cinfo;
	struct jpeg_error_mgr _djerr;
	struct jpeg_error_mgr _cjerr;
	size_t _width, _height, _channels;
	uint8_t* _buffer;
	JSAMPARRAY _scanlines;
};
