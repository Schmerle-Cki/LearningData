#include "mesh.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <sstream>

bool Mesh::intersect(const Ray& r, Hit& h, float tmin,int& objIndex, IntersectMode mode) {

    // Optional: Change this brute force method into a faster one.
    /*bool result = false;
	for (Triangle tri : trigs)
		result |= tri.intersect(r, h, tmin, objIndex, mode);
    return result;*/
	//八叉树求交
	return Tree.intersect(r, h, tmin, objIndex, mode);
}

bool Mesh::interID(const int& triId, const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode)
{
	Triangle tri = trigs[triId];
	return tri.intersect(r, h, tmin, objIndex, mode);
}

Mesh::Mesh(const char *filename, Material *material,const char* textfile) : Object3D(material,textfile) {

    // Optional: Use tiny obj loader to replace this simple one.
    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    char bslash = '/', space = ' ';
    std::string tok;
    int texID;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {		//返回引用
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            Vector3f vec;
            ss >> vec[0] >> vec[1] >> vec[2];
            v.push_back(vec);
        } else if (tok == fTok) {
            if (line.find(bslash) != std::string::npos) {
                std::replace(line.begin(), line.end(), bslash, space);
                std::stringstream facess(line);
                TriangleIndex trig;
                facess >> tok;
                for (int ii = 0; ii < 3; ii++) {
                    facess >> trig[ii] >> texID;
                    trig[ii]--;
                }
                t.push_back(trig);
            } else {
                TriangleIndex trig;
                for (int ii = 0; ii < 3; ii++) {
                    ss >> trig[ii];
                    trig[ii]--;
                }
                t.push_back(trig);
            }
        } else if (tok == texTok) {
            Vector2f texcoord;
            ss >> texcoord[0];
            ss >> texcoord[1];
        }
    }
   
    f.close();
	computeNormal();
	printf("mesh size:%d\n", trigs.size());

	printf("Before build octTree\n");
	Tree.build(this);
	printf("Finish Building\n");
}

void Mesh::computeNormal() {
	n.resize(v.size());
	for (int triId = 0; triId < (int)t.size(); ++triId) {
		TriangleIndex& triIndex = t[triId];
		Vector3f a = v[triIndex[1]] - v[triIndex[0]];
		Vector3f b = v[triIndex[2]] - v[triIndex[0]];
		Vector3f normal = Vector3f::cross(a, b).normalized();

		for (int vertexID = 0; vertexID < 3; vertexID++)
		{
			n[triIndex[vertexID]] += normal;
		}
	}

	//各顶点的法向量为所连接平面的均值
	for (int normalIndex = 0; normalIndex < n.size(); normalIndex++)
	{
		n[normalIndex] = n[normalIndex].normalized();
	}


	//加载所有三角片
	for (int triID = 0; triID < t.size(); triID++)
	{
		TriangleIndex& triIndex = t[triID];
		Triangle trig(v[triIndex[0]], v[triIndex[1]], v[triIndex[2]], n[triIndex[0]], n[triIndex[1]], n[triIndex[2]], this->material, nullptr);
		trigs.push_back(trig);
	}
}


