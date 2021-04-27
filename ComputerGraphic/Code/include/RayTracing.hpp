#pragma once
#include<cassert>
#include"group.hpp"
#include"light.hpp"


void RayTracer(const Ray& ray, const int& depth, float weight, Vector3f& color, const Vector3f& backGround, Group*& baseGroup, Light** lights,std::vector<AreaLight*> alights, const int& numLight)
{
	color = Vector3f::ZERO;
	if (weight < Epsilon)return;
	
	Hit hit; int objIndex = 0;
	bool isIntersect = baseGroup->intersect(ray, hit, Epsilon, objIndex, Refract);
	if (!isIntersect)
	{ 
		color = backGround; return; 
	}

	Vector3f hitPoint = ray.pointAtParameter(hit.getT());
	Vector3f dColor = baseGroup->obj(objIndex)->getColor(hitPoint);
	
	if (Vector3f::dot(hit.getNormal(), ray.getDirection()) <= 0)
	{
		//点光源，方向光源
		for (int li = 0; li < numLight; li++)
		{
			Light* light = lights[li];
			Vector3f L, lightColor;
			light->getIllumination(hitPoint, L, lightColor);
			//添加阴影测试线
			Ray testRay = Ray(hitPoint, L);	Hit hitForTest;
			bool inShadow = baseGroup->intersect(testRay, hitForTest, Epsilon, objIndex, TestShadow);
			if (inShadow)
			{
				double hitToLight = (hitPoint - light->getPos()).length();
				assert(L.length() == 1);
				if (hitForTest.getT() < hitToLight and hitForTest.getMaterial()->w_t() < 1)
					continue;
			}
			
			color += hit.getMaterial()->Shade(ray, hit, L, lightColor, dColor);
		}
	

		//面光源
		for (AreaLight* light:alights)
		{
			for (int sampleTime = 0; sampleTime < SampleLight; sampleTime++)
			{
				Vector3f L, lightColor;
				float a = rand() / float(RAND_MAX), b = rand() / float(RAND_MAX);
				light->getIllumination(hitPoint, L, lightColor, a, b);
				//添加阴影测试线
				Ray testRay = Ray(hitPoint, L);	Hit hitForTest;
				bool inShadow = baseGroup->intersect(testRay, hitForTest, Epsilon, objIndex, TestShadow);
				if (inShadow)
				{
					double hitToLight = (hitPoint - light->getPos(a, b)).length();
					assert(L.length() == 1);
					if (hitForTest.getT() < hitToLight and hitForTest.getMaterial()->w_t() < 1)
						continue;
				}
				color += hit.getMaterial()->Shade(ray, hit, L, lightColor, dColor) / float(SampleLight);
			}
		}
	}

	if (1 < depth)
	{
		Vector3f nextDir = Vector3f::ZERO, moreColor = Vector3f::ZERO;
		Material* curMat = hit.getMaterial();
		Vector3f N = hit.getNormal().normalized(), V = -ray.getDirection().normalized();

		//Reflection
		if (curMat->w_r()>0)
		{
			nextDir = 2 * Vector3f::dot(V, N) * N - V;
			assert(nextDir.length() == 1);
			Ray newRay = Ray(hitPoint, nextDir);
			RayTracer(newRay, (depth - 1), weight * curMat->w_r(), moreColor, backGround, baseGroup, lights, alights, numLight);
			color += moreColor * curMat->w_r();
		}

		//Transmission
		if (curMat->w_t() > 0)
		{
			if (curMat->CalcReractDir(N, -V, nextDir))
			{
				Ray newRay = Ray(hitPoint, nextDir);
				RayTracer(newRay, (depth - 1), weight * curMat->w_t(), moreColor, backGround, baseGroup, lights, alights, numLight);
				color += moreColor * curMat->w_t();
			}
		}
	}
}
