#pragma once
#include <gpu.h>
#include <image.h>

class WaveletTransform2D
{
public:
	Image* source;
	Image* destination;
	void run(cudaStream_t stream);
};
