#ifndef OCTREE_HPP
#define OCTREE_HPP

#include <vecmath.h>
#include <vector>
#include "ray.hpp"
#include "hit.hpp"
#include "triangle.hpp"

class Mesh;



struct OctNode
{
	OctNode* child[8];

	OctNode() {
		for (int i = 0; i < 8; ++i) {
			child[i] = nullptr;
		}
	}
	~OctNode() {
		for (int i = 0; i < 8; ++i) {
			delete child[i];
		}
	}

	bool Leaf() {
		return child[0] == nullptr;
	}

	std::vector<int> obj;
};

class Octree
{
public:
	Octree(int level = 8) :
		maxLevel(level)
	{
	}

	void build(Mesh* m);

	bool intersect(const Ray& ray, Hit& h, float tmin, int& objIndex, IntersectMode mode);

private:
	void buildNode(OctNode* parent, const Box& pbox, const std::vector<int>& trigs, const Mesh& m, int level);

	bool proc_subtree(float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, OctNode* node, const Ray& ray, Hit& h, float tmin, int& objIndex, IntersectMode mode);

	// if a node contains more than 7 triangles and it 
	// hasn't reached the max level yet, split
	static const int max_trig = 7;

	int maxLevel;
	Mesh* mesh;
	Box box;
	OctNode root;
	uint8_t aa; //记录光线方向
};

#endif