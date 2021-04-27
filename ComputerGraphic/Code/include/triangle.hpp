#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>
#include <iostream>

using namespace std;

//AABB包围盒
struct Box
{
	Vector3f Min, Max;

	Box() {}

	Box(const Vector3f& a, const Vector3f& b) :
		Min(a),
		Max(b)
	{}

	Box(float Minx, float Miny, float Minz,
		float Maxx, float  Maxy, float  Maxz) :
		Min(Vector3f(Minx, Miny, Minz)),
		Max(Vector3f(Maxx, Maxy, Maxz))
	{}
};

class Triangle : public Object3D
{

public:
	Triangle() = delete;
	///@param a b c are three vertex positions of the triangle

	Triangle(const Vector3f& a, const Vector3f& b, const Vector3f& c,
		const Vector3f& na, const Vector3f& nb, const Vector3f& nc, Material* m, const char* textfile = nullptr) : Object3D(m, textfile) {
		vertices[0] = a; normals[0] = na;
		vertices[1] = b; normals[1] = nb;
		vertices[2] = c; normals[2] = nc;
		//compute normal
		this->normal = Vector3f::cross((b - a), (c - a)).normalized();
		getBox();
	}

	bool intersect(const Ray& ray, Hit& hit, float tmin,int& objIndex ,IntersectMode mode = Simple) override {
		/*float _d = Vector3f::dot(vertices[0], this->normal);

		//determine the plane of the triangle
		float denominator = Vector3f::dot(ray.getDirection(), this->normal);
		if (denominator == 0)
			return false;

		float t = (_d - Vector3f::dot(ray.getOrigin(), this->normal)) / denominator;
		if (t < tmin || hit.getT() <= t || t < Epsilon)
			return false;

		//internal or external
		Vector3f p = ray.pointAtParameter(t);
		if (Vector3f::dot(Vector3f::cross((vertices[1] - vertices[0]), (p - vertices[0])), this->normal) < 0)
			return false;
		if (Vector3f::dot(Vector3f::cross((vertices[0] - vertices[2]), (p - vertices[2])), this->normal) < 0)
			return false;
		if (Vector3f::dot(Vector3f::cross((vertices[2] - vertices[1]), (p - vertices[1])), this->normal) < 0)
			return false;

		hit.set(t, this->material, this->normal);
		return true;*/
		Matrix3f Mtx(vertices[0] - vertices[1], vertices[0] - vertices[2], ray.getDirection());
		Vector3f RandVec = vertices[0] - ray.getOrigin();
		Vector3f Ratio = Mtx.inverse() * RandVec;

		//计算交点的顶点表示
		float alpha = 1 - Ratio[0] - Ratio[1];
		float beta = Ratio[0];
		float theta = Ratio[1];

		float t = Ratio[2];
		if (t > hit.getT() || t < tmin || alpha < 0 || beta < 0 || theta < 0)return false;

		Vector3f hitNormal = (alpha * normals[0] + beta * normals[1] + theta * normals[2]).normalized();
		hit.set(t, this->material, hitNormal);
		return true;
	}

	const Vector3f& getVertex(int index) const
	{
		//printf("calling me!\n");
		assert(index < 3);
		return vertices[index];
	}

	const Vector3f& getNormal(int index) const
	{
		assert(index < 3);
		return normals[index];
	}

	//三角形包围盒大小确定
	void getBox()
	{
		AABB.Min = vertices[0];
		AABB.Max = vertices[0];

		for (int ii = 1; ii < 3; ii++) {
			for (int dim = 0; dim < 3; dim++) {
				if (AABB.Min[dim] > vertices[ii][dim]) {
					AABB.Min[dim] = vertices[ii][dim];
				}
				if (AABB.Max[dim] < vertices[ii][dim]) {
					AABB.Max[dim] = vertices[ii][dim];
				}
			}
		}
	}

	Vector3f normal;
	Vector3f vertices[3];
	Vector3f normals[3];
	Box AABB;
protected:
};

#endif //TRIANGLE_H
