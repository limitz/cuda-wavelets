#include <gpu.h>
#include <image.h>
#include <display.h>

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

		Image* img = Image::load("kodak.ppm");
		img->toDevice(stream);
		//img->convert(Image::ColorSpace::Grayscale, stream);
		img->convert(Image::ColorSpace::sRGB, stream);
		//img->convert(Image::ColorSpace::Device, stream);
		img->printInfo();

		while (true)
		{
			display.render(img);
			
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
