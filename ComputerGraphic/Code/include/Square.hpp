#ifndef SQUARE_H
#define SQUARE_H

#include "triangle.hpp"

class Square : public Object3D
{

public:
	Square() = delete;
	///@param a b c are three vertex positions of the triangle

	Square(const Vector3f& A, const Vector3f& B, const Vector3f& C,
		const Vector3f& D, const Vector3f& n, Material* m, const char* textfile = nullptr) : Object3D(m, textfile), tri1(A, B, C, n, n, n, m), tri2(A, C, D, n, n, n, m) {
		vertices[0] = C;
		vertices[1] = B - C; vertices[2] = D - C;
		h = vertices[1].length(); w = vertices[2].length();
		vertices[1] = vertices[1] / h; vertices[2] = vertices[2] / w;
		
	}

	bool intersect(const Ray& ray, Hit& hit, float tmin, int& objIndex, IntersectMode mode = Simple) override {
		if (tri1.intersect(ray, hit, tmin, objIndex, mode))
			return true;
		return tri2.intersect(ray, hit, tmin, objIndex, mode);
	}

	Vector3f getColor(const Vector3f P)const override
	{
		if (this->img == nullptr)return Vector3f(MAX_FLOAT);

		Vector3f CP = P - vertices[0];
		
		float xx = abs(Vector3f::dot(CP, vertices[2])) / w;
		if (xx < Epsilon)xx += Epsilon;
		if (abs(xx - 1) < Epsilon)xx -= Epsilon;
		int x = xx * this->img->Width();
		if (x < 0)x = 0; 
		if(x >= img->Width())
		{
			printf("oleegal x!%d\n", x);
			x = img->Width() - 1;
		}

		float yy = abs(Vector3f::dot(CP, vertices[1])) / h;
		if (yy < Epsilon)yy += Epsilon;
		if (abs(yy - 1) < Epsilon)yy -= Epsilon;
		int y = yy * this->img->Height();

		return this->img->GetPixel(x, y);
		
	}

	Triangle tri1;
	Triangle tri2;
	Vector3f vertices[3];
	float w, h;
protected:
};

#endif //TRIANGLE_H

