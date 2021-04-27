#ifndef SPHERE_H
#define SPHERE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>


class Sphere : public Object3D {
public:
	Sphere() {
		// unit ball at the center
		center = Vector3f::ZERO;
		radius = 1;
	}

	Sphere(const Vector3f& ccenter, float rradius, Material* material,const char* textfile=nullptr) : Object3D(material,textfile), center(ccenter), radius(rradius) {
		//
	}

	~Sphere() override = default;

	float Radius() const {
		return radius;
	}

	Vector3f getColor(const Vector3f P) const override {
		return this->addTexture(P - this->center);
	}

	Vector3f addTexture(Vector3f CP) const
	{
		if (this->img == nullptr)return Vector3f(MAX_FLOAT);

		if (CP[1] == 0 && CP[2] == 0)return img->GetPixel(0, 0);
		if (abs(CP[2] - radius) < Epsilon)CP[2] -= Epsilon;
		if (abs(CP[2] + radius) < Epsilon)CP[2] += Epsilon;

		float theta = acos(CP[2] / this->radius);
		float phi = atan2(CP[1], CP[0]) + M_PI;
		int x = phi / (2 * M_PI) * img->Width();
		int y = theta / M_PI * img->Height();

		return this->img->GetPixel(x, y);

	}

	bool intersect(const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode = Simple) override {
		Vector3f OC = center - r.getOrigin();
		float OH_len = Vector3f::dot(OC, r.getDirection()) / r.getDirection().length();

		if (OH_len <= 0)
			return false;

		float squaredCH = OC.squaredLength() - OH_len * OH_len;

		float squaredR = radius * radius;
		if (squaredR < squaredCH)
			return false;

		float PH_len = sqrt(squaredR - squaredCH);
		float t = (OH_len - PH_len) / r.getDirection().length();
		//origin is one the sphere
		if (t < Epsilon) 
		{
			if (mode == Simple)
			{
				return false;
			}
			else if (mode == Refract)
			{
				t = (OH_len + PH_len) / r.getDirection().length();
			}
			else
			{
				t = (OH_len + PH_len) / r.getDirection().length();
				
			}
		}

		if (t < tmin || h.getT() <= t)
			return false;

		Vector3f P = r.pointAtParameter(t);
		Vector3f normal = (P - center).normalized();
		h.set(t, this->material, normal);
		return true;
	}

protected:
	Vector3f center;
	float radius;

};


#endif
