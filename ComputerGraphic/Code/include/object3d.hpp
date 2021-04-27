#ifndef OBJECT3D_H
#define OBJECT3D_H


#include "ray.hpp"
#include "hit.hpp"
#include "material.hpp"
#include "Constants.h"
#include "image.hpp"

// Base class for all 3d entities.
class Object3D {
public:
    Object3D() : material(nullptr) {}

    virtual ~Object3D() = default;

	virtual Vector3f getColor(const Vector3f P) const
	{
		return Vector3f(50.0);
	}

	explicit Object3D(Material* material, const char* textFile = nullptr) {
        this->material = material;
		img = (textFile == nullptr ? nullptr : Image::LoadImage(textFile));
    }

    // Intersect Ray with this object. If hit, store information in hit structure.
	virtual bool intersect(const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode = Simple) = 0;
	
protected:

    Material *material;
	Image* img;
};

#endif

