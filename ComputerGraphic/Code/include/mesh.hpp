#ifndef MESH_H
#define MESH_H

#include <vector>
#include "object3d.hpp"
#include "triangle.hpp"
#include "Vector2f.h"
#include "Vector3f.h"
#include "octTree.h"

//网格
class Mesh : public Object3D {

public:
	Mesh(const char* filename, Material* m, const char* textfile = nullptr);

    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0; x[1] = 0; x[2] = 0;  //三角形三个顶点在v中的编号
        }
        int &operator[](const int i) { return x[i]; }
        // By Computer Graphics convention, counterclockwise winding is front face（按照计算机图形学的惯例，逆时针缠绕是正面）
        int x[3]{};
    };

    std::vector<Vector3f> v;   //顶点集
    std::vector<TriangleIndex> t;	//三角面片集
    std::vector<Vector3f> n;
	std::vector<Triangle> trigs;
	bool intersect(const Ray& r, Hit& h, float tmin,int& objIndex ,IntersectMode mode = Simple) override;
	bool interID(const int& triId, const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode = Simple);
	const vector<Triangle> getTriangles() const
	{
		return trigs;
	}

private:

    // Normal can be used for light estimation
    void computeNormal();
	Octree Tree;
};

#endif
