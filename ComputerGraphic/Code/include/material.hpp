#ifndef MATERIAL_H
#define MATERIAL_H

#include <cassert>
#include <vecmath.h>

#include "hit.hpp"
#include <iostream>
#include <cmath>


class Material {
public:

    explicit Material(const Vector3f &d_color, const Vector3f &s_color = Vector3f::ZERO, float s = 0,float w_r = 0,float w_t=0,float refra=1.0) :
            diffuseColor(d_color), specularColor(s_color), shininess(s),w_reflect(w_r),w_transmit(w_t),Refraction(refra) {
    }

    virtual ~Material() = default;

    virtual Vector3f getDiffuseColor() const {
        return diffuseColor;
    }
	
	virtual float w_r() const
	{
		return w_reflect;
	}

	virtual float w_t() const
	{
		return w_transmit;
	}

	virtual float refractIndex() const {
		return Refraction;
	}

	Vector3f Shade(const Ray& ray, const Hit& hit,//t+material+normal
		const Vector3f& dirToLight, const Vector3f& lightColor, Vector3f dColor) {
        Vector3f shaded = Vector3f::ZERO;
		Vector3f L = dirToLight, N = hit.getNormal(), R = 2 * Vector3f::dot(L, N) * N - L, V = -ray.getDirection();
		
		float RELU1 = Vector3f::dot(L, N), RELU2 = Vector3f::dot(V, R);
		RELU1 = (RELU1 < 0 ? 0 : RELU1);
		RELU2 = (RELU2 < 0 ? 0 : RELU2);
		
		if (dColor[0] > 1.0)
		{
			dColor = diffuseColor;
		}
		shaded = lightColor * (dColor * RELU1 + this->specularColor * pow(RELU2, this->shininess));
        return shaded;
    }

	bool CalcReractDir(const Vector3f& normal, const Vector3f& indir, Vector3f& outdir)
	{
		if (Refraction == 1.0)
		{
			outdir = indir;
			return true;
		}

		Vector3f N = normal, V = indir;
		assert(N.length() == 1 and V.length() == 1);
		float cos1 = Vector3f::dot(N, V);
		float eta = cos1 < 0.0 ? 1.0f / Refraction : Refraction; 

		float sin2Sq = (1.0 - cos1 * cos1) * eta * eta;
		if (sin2Sq > 1.0)return false;
		float cos2 = sqrt(1.0 - sin2Sq);

		outdir = eta * (V + N * cos1) - cos2 * N;
		outdir = outdir.normalized();
		return true;
	}

protected:
    Vector3f diffuseColor;		//漫反射
    Vector3f specularColor;		//镜面反射
    float shininess;			//反射指数n

	float w_reflect;			//反射衰减系数
	float w_transmit;			//折射衰减系数
	float Refraction;			//折射率
};


#endif // MATERIAL_H
