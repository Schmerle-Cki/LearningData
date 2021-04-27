#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

class Plane : public Object3D {
public:
	Plane() {

	}

	Plane(const Vector3f& normal, float dd, Material* m,const char* textfile = nullptr) : Object3D(m,textfile) {
		this->norm = normal;
		this->d = dd;
	}

	~Plane() override = default;

	bool intersect(const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode = Simple) override {
		float denominator = Vector3f::dot(r.getDirection(), this->norm);

		if (denominator == 0)
			return false;

		float t = (this->d - Vector3f::dot(r.getOrigin(), this->norm)) / denominator;
		if (t < tmin || h.getT() <= t || t < Epsilon)return false;


		h.set(t, this->material, this->norm);
		return true;
	}

protected:
	Vector3f norm;
	float d;

};

#endif //PLANE_H


