#include <gpu.h>
#include <image.h>
#include <display.h>
#include <wavelet.h>

#define ESCAPE 27

int main(int /*argc*/, char** /*argv*/)
{
	int rc;
	try 
	{
		selectGPU();
		
		cudaStream_t stream = 0;
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";
		
		// SETUP DISPLAY
		CudaDisplay display("Wavelets", 1920, 1080); 

		Image::Default.stream = stream;
		Image::Default.width = 1920;
		Image::Default.height = 1080;
		Image::Default.channels = 3;
		Image::Default.colorSpace = ColorSpace::Device;

		Image testImage("kodak.ppm");
		testImage.width = Image::Default.width;
		testImage.height = Image::Default.height;

		Image transformed(ColorSpace::Default, Image::Default.width, Image::Default.height, Image::Default.channels);

		WaveletTransform2D wt;
		wt.source = &testImage;
		wt.destination = &transformed;
		wt.run(stream);
		while (true)
		{
			display.render(wt.destination);
			
			while (int e = display.events())
			{
				switch (e)
				{
					case 'q':
					case ESCAPE:
						cudaStreamDestroy(stream);
						return 0;
					
					default:break;
				}
			}
			usleep(100000);
		}
	}
	catch (const char* &ex)
	{
		fprintf(stderr, "ERROR: %s\n", ex);
		fflush(stderr);
	 	return 1;
	}
	
	return 0;
}
