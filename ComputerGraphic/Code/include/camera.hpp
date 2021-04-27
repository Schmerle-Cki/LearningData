#define _CRT_SECURE_NO_WARNINGS
#ifndef CAMERA_H
#define CAMERA_H

#include "ray.hpp"
#include "Constants.h"
#include <vecmath.h>
#include <float.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>


class Camera {
public:
    Camera(const Vector3f &center, const Vector3f &direction, const Vector3f &up, int imgW, int imgH) {
        this->center = center;
        this->direction = direction.normalized();
        this->horizontal = Vector3f::cross(this->direction, up);
        this->up = Vector3f::cross(this->horizontal, this->direction);  //使得this->up与this->direction、this->horizontal正交
        this->width = imgW;
        this->height = imgH;
    }

    // Generate rays for each screen-space coordinate
    virtual Ray generateRay(const Vector2f &point) = 0;
    virtual ~Camera() = default;

    int getWidth() const { return width; }
    int getHeight() const { return height; }

protected:
    // Extrinsic parameters
    Vector3f center;
    Vector3f direction;
    Vector3f up;
    Vector3f horizontal;
    // Intrinsic parameters
    int width;
    int height;
};

// TODO: Implement Perspective camera
// You can add new functions or variables whenever needed.
class PerspectiveCamera : public Camera {

public:
	PerspectiveCamera(const Vector3f& center, const Vector3f& direction,
		const Vector3f& up, int imgW, int imgH, float angle) : Camera(center, direction, up, imgW, imgH), _angle(angle) {
		OpticalCenter = Vector2f(this->width / 2, this->height / 2);
		Rotate = Matrix3f(this->horizontal.normalized(), this->up.normalized(), this->direction.normalized());

        // angle is in radian.
		float cameraRule = tan(_angle / 2);
		fx = (this->width / 2) / cameraRule;
		fy = (this->height / 2) / cameraRule;
    }

    Ray generateRay(const Vector2f &point) override {
        // 
		Vector3f Rc = Vector3f((point[0] - this->OpticalCenter[0]) / this->fx, (point[1] - this->OpticalCenter[1]) / this->fy, 1);
		Rc.normalize();
		Vector3f Rw = Rotate * Rc;

		float rand1 = 2 * (rand() / float(RAND_MAX)) - 1, rand2 = 2 * (rand() / float(RAND_MAX)) - 1;
		Vector3f offset(rand1, rand2, 0);
		float costheta = abs(Vector3f::dot(this->direction, Rw));
		if (costheta < Epsilon)costheta = Epsilon;
		Vector3f hitPoint = this->center + Rw * Focus_Dis / costheta;
		Vector3f origin = this->center + offset * aperture / 2;
		Rw = (hitPoint - origin).normalized();

		return Ray(origin, Rw);
		//return Ray(this->center, Rw);
    }
private:
	Vector2f OpticalCenter;
	Matrix3f Rotate;
	float _angle;
	float fx, fy;
};

#endif //CAMERA_H
