#ifndef CURVE_HPP
#define CURVE_HPP

#include "object3d.hpp"
#include <vecmath.h>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>
using namespace std;


// The CurvePoint object stores information about a point on a curve
// after it has been tesselated: the vertex (V) and the tangent (T)
// It is the responsiblility of functions that create these objects to fill in all the data.
struct CurvePoint {
    Vector3f V; // Vertex
    Vector3f T; // Tangent  (unit)切向
};

class Curve : public Object3D {
protected:
    std::vector<Vector3f> controls;
public:
    explicit Curve(std::vector<Vector3f> points,Material* m,const char* textfile=nullptr) :Object3D(m,textfile), controls(std::move(points)) {}

	bool intersect(const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode = Simple) override {
        return false;
    }

    std::vector<Vector3f> &getControls() {
        return controls;
    }

    virtual void discretize(int resolution, std::vector<CurvePoint>& data) = 0;

    /*void drawGL() {
        Object3D::drawGL();
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 0);
        glBegin(GL_LINE_STRIP);
        for (auto & control : controls) { glVertex3fv(control); }
        glEnd();
        glPointSize(4);
        glBegin(GL_POINTS);
        for (auto & control : controls) { glVertex3fv(control); }
        glEnd();
        std::vector<CurvePoint> sampledPoints;
        discretize(30, sampledPoints);
        glColor3f(1, 1, 1);
        glBegin(GL_LINE_STRIP);
		for (auto& cp : sampledPoints) glVertex3fv(cp.V); 
        glEnd();
        glPopAttrib();
    }*/
};

class BezierCurve : public Curve {
public:
	//保证控制点位于XY平面的第一象限
	explicit BezierCurve(const std::vector<Vector3f>& points, Material* m,const char* textfile = nullptr) : Curve(points, m,textfile) {
        if (points.size() < 4 || points.size() % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
            exit(0);
        }
		ratioThird = -1 * controls[0] + 3 * controls[1] - 3 * controls[2] + controls[3];
		ratioSecnd = 3 * controls[0] - 6 * controls[1] + 3 * controls[2];
		ratioFirst = -3 * controls[0] + 3 * controls[1];
		ratioZero = controls[0];

		dr2 = -3 * controls[0] + 9 * controls[1] - 9 * controls[2] + 3 * controls[3];
		dr1 = 6 * controls[0] - 12 * controls[1] + 6 * controls[2];
		dr0 = -3 * controls[0] + 3 * controls[1];

		maxX = maxY = -1; minY = MAX_FLOAT;
		for (auto point : controls)
		{
			if (point[0] > maxX)maxX = point[0];
			if (point[1] > maxY)maxY = point[1];
			if (point[1] < minY)minY = point[1];
		}
    }

    void discretize(int resolution, std::vector<CurvePoint>& data) override {
        data.clear();
		float step = 1.0 / resolution;
		for (int i = 1; i < resolution; i++)
		{
			CurvePoint p;
			p.V = f(i * step);
			p.T = df(i * step).normalized();
			data.push_back(p);
		}
    }

	Vector3f f(float t)
	{
		return ratioZero + ratioFirst * t + ratioSecnd * t * t + ratioThird * t * t * t;
	}

	Vector3f df(float t)
	{
		return dr2 * t * t + dr1 * t + dr0;
	}

	Vector3f ddf(float t)
	{
		return 2 * dr2 * t + dr1;
	}

	bool intersect(const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode= Simple) override {
		float A = 0.0, B = 0.0, C = 0.0, t, u, theta;
		Vector3f finalNormal = Vector3f::UP, ori = r.getOrigin(), dir = r.getDirection();
		int res = getABC(r, A, B, C);
		
		if (res == 1)
		{	
			if (B < 0 || B > maxY)return false;

			int resolution = 10;
			if (abs(dir[1]) < 0.06)
			{
				resolution = 40;
			}

			u = NewTon_F(A, B, C, r, resolution);
			if (u < 0 || u > 1 || f(u)[0] > maxX)return false;

			t = (f(u)[1] - r.getOrigin()[1]) / r.getDirection()[1];
			if (t >= h.getT() || t < tmin)return false;		
		}
		else if (res == 0)
		{
			if (r.getDirection()[1] == 1)
			{
				t = minY - r.getOrigin()[1];
				if (t >= h.getT() || t < tmin)return false;
				finalNormal = Vector3f(0, 0, -1);
				h.set(t, this->material, finalNormal);
				return true;
			}
			else
			{
				t = -maxY + r.getOrigin()[1];
				if (t >= h.getT() || t < tmin)return false;
				finalNormal = Vector3f(0, 0, 1);
				h.set(t, this->material, finalNormal);
				return true;
			}
		}
		else
		{
			
			if (mode == TestShadow)return false;

			float Y = r.getOrigin()[1];
			if (Y<minY || Y>maxY)return false;

			u = NewTon_Y(Y);
			if (u < 0 || u > 1)return false;

			Vector3f center(0, Y, 0);
			float radius = f(u)[0];
			assert(radius > 0);

			Vector3f OC = center - r.getOrigin();
			float OH_len = Vector3f::dot(OC, r.getDirection()) / r.getDirection().length();

			if (OH_len <= 0)
				return false;

			float squaredCH = OC.squaredLength() - OH_len * OH_len;

			float squaredR = radius * radius;
			if (squaredR < squaredCH)
				return false;

			float PH_len = sqrt(squaredR - squaredCH);
			t = (OH_len - PH_len) / r.getDirection().length();
			
			if (t < tmin || h.getT() <= t)
				return false;
		}
		
		Vector3f P = r.pointAtParameter(t);
		theta = atan2(P[2], P[0]);
		if (theta < 0)theta += 2 * M_PI;
		Vector3f v0(cos(theta), 0, sin(theta)), v1(0, 1, 0), v2(-sin(theta), 0, cos(theta)), T = df(u).normalized();
		Matrix3f M(v0, v1, v2, 1);
		Vector3f pn = Vector3f::cross(T, -Vector3f::FORWARD);
		finalNormal = M * pn;
		h.set(t, this->material, finalNormal);
		return true;
	}

	//牛顿迭代法求近似解
	float NewTon_F(const float& A, const float& B, const float& C, const Ray& r,const int& resolution)
	{
		float interval = 1.0 / float(resolution), ft, dft, ddft, yt, xt, dy, dx;
		for (float up = interval; up <= 1; up += interval) {
			float low = up - interval, t = (low + up) / 2;
			for (int i = 10; i--; )
			{
				if (t < 0) t = low;
				else if (t > 1) t = up;

				yt = f(t)[1]; xt = f(t)[0]; dy = df(t)[1]; dx = df(t)[0];

				float inSquare = A * (yt - B) * (yt - B) + C;
				assert(inSquare > 0);
				ft = sqrt(inSquare) - xt;
				dft = A * (yt - B) * dy / (ft + xt) - dx;

				if (std::abs(ft) < 0.01)
				{
					float u = (f(t)[1] - r.getOrigin()[1]) / r.getDirection()[1];
					if (checkabT(r, u))
						return t;
					else
					{
						break;
					}
				}
				assert(dft != 0);
				t += -ft / dft;
			}
		}
		return -1;
	}

	float NewTon_alpha(const float& A, const float& B, const float& C, const float& xk, const float& dk)
	{
		float alpha = 1, ft, dft, ddft, y, x, dy, dx, ddy, ddx;
		for (int i = 7; i--; )
		{
			float t = xk + alpha * dk;

			y = f(t)[1]; x = f(t)[0]; dy = df(t)[1]; dx = df(t)[0]; ddy = ddf(t)[1]; ddx = ddf(t)[0];

			float inSquare = A * (y - B) * (y - B) + C;
			assert(inSquare >= 0);
			ft = sqrt(inSquare) - x;
			dft = (A * (y - B) * dy / (ft + x) - dx) * dk;
			ddft = dk * dk * (A * (A * (y - B) * (y - B) * (y - B) * ddy + C * (dy * dy + y * ddy - B * ddy)) / (inSquare * (ft + x)) - ddx);

			if (std::abs(dft) < 0.01)
			{
				return alpha;
			}
			assert(ddft != 0);
			alpha -= dft / ddft;
		}
		return 1.0;
	}

	float NewTon_Y(const float& Y)
	{
		float t = (Y - minY) / (maxY - minY), ft, dft;
		{
			for (int i = 15; i--; )
			{
				if (t < 0) t = rand() / float(MAX_FLOAT);
				else if (t > 1) t = rand() / float(MAX_FLOAT);

				ft = f(t)[1] - Y; dft = df(t)[1];

				if (std::abs(ft) < 0.01)
					return t;
				assert(dft != 0);
				t -= ft / dft;
			}
		}
		return -1;
	}

	bool checkabT(const Ray& r,const float& t)
	{
		return (t > Epsilon && r.pointAtParameter(t)[2] * r.getDirection()[2] <= 0);
	}

	int getABC(const Ray& ray, float& A, float& B, float& C)
	{
		Vector3f o = ray.getOrigin(), d = ray.getDirection();
		
		if (d[1] == 1 || d[1] == -1)
		{
			return 0;
		}
		else if (abs(d[1]) < 0.008)
		{
			return -1;
		}
		else
		{
			float d0_d2 = d[0] * d[0] + d[2] * d[2];
			A = d0_d2 / (d[1] * d[1]);
			B = o[1] - d[1] * (o[0] * d[0] + o[2] * d[2]) / d0_d2;
			C = (o[2] * d[0] - o[0] * d[2]) * (o[2] * d[0] - o[0] * d[2]) / d0_d2;
			return 1;
		}
	}

	Vector3f addTexture(const Vector3f P) const
	{
		if (this->img == nullptr)return Vector3f(MAX_FLOAT);

		
		float theta = atan2(P[2], P[0]) + M_PI;
		int x = theta / (2 * M_PI) * this->img->Width();
		
		float yy = (P[1] - minY) / (maxY - minY); 
		if (P[1] == 1)printf("me!\n");
		if (abs(yy) < Epsilon)yy += Epsilon;
		if (abs(yy - 1) < Epsilon)yy -= Epsilon;
		int y = yy * this->img->Height();
		if (y < 0 || y >= this->img->Height())
		{
			printf("ileegl y!Y:%f\n",P[1]);
			y = 0;
		}

		return this->img->GetPixel(x, y);
	}

	Vector3f getColor(const Vector3f P) const override {
		return this->addTexture(P);
	}

protected:
	Vector3f ratioThird;
	Vector3f ratioSecnd;
	Vector3f ratioFirst;
	Vector3f ratioZero;

	Vector3f dr0;
	Vector3f dr1;
	Vector3f dr2;

	float maxX;
	float maxY;
	float minY;
};




#endif // CURVE_HPP
