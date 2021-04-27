#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "omp.h"

#include "scene_parser.hpp"
#include "image.hpp"
#include "camera.hpp"
#include "group.hpp"
#include "light.hpp"
#include "RayTracing.hpp"

#include <string>
using namespace std;

int main(int argc, char *argv[]) {
	double enterTime = clock();
    for (int argNum = 1; argNum < argc; ++argNum) {
        std::cout << "Argument " << argNum << " is: " << argv[argNum] << std::endl;
    }

    if (argc != 3) {
        std::cout << "Usage: ./bin/PA1 <input scene file> <output bmp file>" << std::endl;
        return 1;
    }
    string inputFile = argv[1];
    string outputFile = argv[2];  // only bmp is allowed.
	
	
   
	SceneParser sceneParser(inputFile.data());
	Camera* camera = sceneParser.getCamera();
	Image image(camera->getWidth(), camera->getHeight());
	Image normalMap(camera->getWidth(), camera->getHeight());

	srand(int(time(NULL)));
	double prePare = (clock() - enterTime) / CLOCKS_PER_SEC;
	cout << "Prepared Time:" << prePare << endl;
	double start_time = clock(), last = start_time;
#pragma omp parallel for schedule(dynamic) private(ray)
	for (int x = 0; x < camera->getWidth(); ++x) {
		for (int y = 0; y < camera->getHeight(); ++y)
		{
			double interval = (clock() - last) / CLOCKS_PER_SEC;
			if (interval > 10)
			{
				printf("ThreadID:%d:I'm running!%f\t(%d,%d)\n", 0, float(interval), x, y);
				last = clock();
				interval = 0;
			}
			Group* baseGroup = sceneParser.getGroup();
			Light** lights = sceneParser.getLights();
			int lightNum = sceneParser.getNumLights();
			vector<AreaLight*> alights = sceneParser.getArealights();
			Vector3f finalColor = Vector3f::ZERO;
			//Vector3f fianlNormal = Vector3f::ZERO;
			for (int sy = 0; sy < 2; ++sy)				//将各像素细分为四个子像素
				for (int sx = 0; sx < 2; ++sx)
				{
					Vector3f subColor = Vector3f::ZERO;	//超采样，抗锯齿
					for (int s = 0; s < SampleCount; ++s) {
						float r1 = 2.0f * rand()/float(RAND_MAX), dx = r1 < 1 ? sqrt(r1)-1 : 2 - sqrt(2 - r1); //随机数tent滤波，将【0,2】区间映射到【-1,1】
						float r2 = 2.0f * rand()/float(RAND_MAX), dy = r2 < 1 ? sqrt(r2)-1 : 2 - sqrt(2 - r2);
						Ray ray = camera->generateRay(Vector2f((sx + dx + 0.5) / 2 + x, (sy + dy + 0.5) / 2 + y));
						Vector3f newColor = Vector3f::ZERO;
						RayTracer(ray, MAX_DEPTH, 1.0, newColor, sceneParser.getBackgroundColor(), baseGroup, lights, alights, lightNum);
						subColor += newColor * (1.0 / SampleCount);
					}
					finalColor += subColor * 0.25;
				}
			image.SetPixel(x, y, finalColor);
		}
	}
	double usedTime = (clock() - start_time) / CLOCKS_PER_SEC;
	cout << "Total Render Time:" << usedTime << endl;

	image.SaveImage(outputFile.data());
    std::cout << "Hello! Computer Graphics!" << std::endl;
    return 0;
}

