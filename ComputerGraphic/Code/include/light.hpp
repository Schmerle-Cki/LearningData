#ifndef LIGHT_H
#define LIGHT_H

#include <Vector3f.h>
#include "object3d.hpp"

class Light {
public:
    Light() = default;

    virtual ~Light() = default;

    virtual void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const = 0;
	virtual Vector3f getPos() const = 0;
};


class DirectionalLight : public Light {
public:
    DirectionalLight() = delete;

    DirectionalLight(const Vector3f &d, const Vector3f &c) {
        direction = d.normalized();
        color = c;
    }

    ~DirectionalLight() override = default;

    ///@param p unsed in this function
    ///@param distanceToLight not well defined because it's not a point light
    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = -direction;
        col = color;
    }

	Vector3f getPos() const override
	{
		return Vector3f(MAX_FLOAT);
	}

private:

    Vector3f direction;
    Vector3f color;

};

class PointLight : public Light {
public:
    PointLight() = delete;

    PointLight(const Vector3f &p, const Vector3f &c) {
        position = p;
        color = c;
    }

    ~PointLight() override = default;

    void getIllumination(const Vector3f &p, Vector3f &dir, Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = (position - p);
        dir = dir / dir.length(); //单位化方向
        col = color;
    }

	Vector3f getPos() const override
	{
		return position;
	}

private:

    Vector3f position;
    Vector3f color;

};

class AreaLight{
public:
	AreaLight() = delete;

	AreaLight(const Vector3f& c, const Vector3f& n,const Vector3f& co ,const float& r) {
		center = c; planeNormal = n.normalized(); radius = r; color = co;
		bias = Vector3f::dot(center, planeNormal);
		Vector3f P = Vector3f::ZERO;
		if (n.x() != 0)P.x() = bias / n.x();
		else if (n.y() != 0)P.y() = bias / n.y();
		else P.z() = bias / n.z();

		Vector3f CP = P - center;
		assert(!(CP == Vector3f::ZERO));
		fixX = CP / CP.length();
		fixY = Vector3f::cross(planeNormal, CP).normalized();
	}

	~AreaLight()  = default;

	///@param p unsed in this function
	///@param distanceToLight not well defined because it's not a point light
	void getIllumination(const Vector3f & p, Vector3f & dir, Vector3f & col,float rand1,float rand2) const  {
		Vector3f pos = getPos(rand1, rand2);
		dir = (pos - p).normalized();
		col = color;
	}

	Vector3f getPos(float rand1, float rand2) const
	{
		float r = rand1 * radius;
		float theta = rand2 * 2 * M_PI;

		rand1 = cos(theta);
		rand2 = sin(theta);
		Vector3f CP = rand1 * fixX + rand2 * fixY;
		return center + CP * r;
	}


private:

	Vector3f color;
	Vector3f center;
	Vector3f planeNormal;
	Vector3f fixX;
	Vector3f fixY;
	float radius;
	float bias;

};

#endif // LIGHT_H
