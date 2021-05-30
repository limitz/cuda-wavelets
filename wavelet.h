#pragma once
#include <gpu.h>
#include <image.h>

class Wavelet
{
public:
	Wavelet(const char* name);
	~Wavelet();

	void init(const float* mother, size_t len);

	size_t length;

	struct
	{
		struct
		{
			float* device;
			float* host;
		} H,G;
	} decompose, reconstruct;

};

class WaveletTransform2D
{
public:
	size_t levels = 1;
	Image* source = nullptr;
	Image* destination = nullptr;
	Image* intermediate1 = nullptr;
	Image* intermediate2 = nullptr;
	Wavelet* wavelet = nullptr;
	void run(cudaStream_t stream);
};
