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

class Image
{
public:
	enum ColorSpace
	{
		Device,
		Grayscale,
		sRGB,
		CIELab
	};

	size_t width, height, channels;
	const char* filename;
	ColorSpace colorSpace = ColorSpace::Device;

	struct
	{
		struct
		{
			float4* data;
			size_t pitch;
		} host, device;
	} mem;

	~Image();

	static Image* create(size_t width, size_t height, size_t channels = 0);
	static Image* load(const char* filename = nullptr);
	static Image* save(const char* filename = nullptr);

	void toDevice(cudaStream_t stream);
	void toHost(cudaStream_t stream);
	
	void convert(ColorSpace cs, cudaStream_t stream);
	void printInfo();

	float psnr(const Image* reference);

private:
	Image();

	void loadPPM();
	void loadPGM();
	void loadJPG();
	char* _filename;

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
