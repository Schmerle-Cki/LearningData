#include "octTree.h"
#include "mesh.hpp"
#include <algorithm>

//线段交叠测试
bool intersect(float* a, float* b)
{
	/*if (a[0] > b[1]) {
		return a[0] <= b[1];
	}
	else {
		return b[0] <= a[1];
	}*/
	if (a[0] > b[1]) {
		float* tmp = a;
		a = b;
		b = tmp;
	}
	return b[0] <= a[1];
}

//包围盒a、b交叠测试
bool boxOverlap(const Box* a,const Box* b)
{
	//在每个维度都有重叠
	for (int dim = 0; dim < 3; dim++) {
		float ia[2] = { a->Min[dim], a->Max[dim] };
		float ib[2] = { b->Min[dim], b->Max[dim] };
		if (!intersect(ia, ib)) {
			return false;
		}
	}
	return true;
}

//a是否被b包围
bool inside(const Box& a, const Box& b)
{
	for (int dim = 0; dim < 3; dim++) {
		if (a.Min[dim] < b.Min[dim] || a.Max[dim] > b.Max[dim]) {
			return false;
		}
	}
	return true;
}

bool PointInside(const Box& b, const Vector3f& p)
{
	return ((b.Min[0] <= p[0]) && (p[0] <= b.Max[0])
		&& (b.Min[1] <= p[1]) && (p[1] <= b.Max[1])
		&& (b.Min[2] <= p[2]) && (p[2] <= b.Max[2]));
}

bool triOverLap(int triId, const Mesh& m,const Box& box)
{
	const auto& tri = m.getTriangles()[triId];
	
	//printf(";;;%d", m.getTriangles().size());
	Vector3f a = tri.getVertex(0);
	//printf("entering buildchild;trigID:%d\n", triId);
	Vector3f b = tri.getVertex(1), c = tri.getVertex(2);
	return ((PointInside(box, a)) || (PointInside(box, b)) || (PointInside(box, c)));
}

//递归式地构造树-O(n)(高度钳制)
void Octree::buildNode(OctNode* parent, const Box& pbox, const std::vector<int>& trigs, const Mesh& m, int level)
{
	if (trigs.size() <= Octree::max_trig || level > maxLevel) {
		parent->obj = trigs;
		return;
	}

	level++;
	//初始化子节点
	for (int childID = 0; childID < 8; childID++) {
		parent->child[childID] = new OctNode();
	}

	const Vector3f& mn = pbox.Min;
	const Vector3f& mx = pbox.Max;
	Vector3f mid = (mn + mx) / 2.0;

	Box cBox[8];
	cBox[0] = Box(mn, mid);
	cBox[1] = Box(mn[0], mn[1], mid[2], mid[0], mid[1], mx[2]);
	cBox[2] = Box(mn[0], mid[1], mn[2], mid[0], mx[1], mid[2]);
	cBox[3] = Box(mn[0], mid[1], mid[2], mid[0], mx[1], mx[2]);
	cBox[4] = Box(mid[0], mn[1], mn[2], mx[0], mid[1], mid[2]);
	cBox[5] = Box(mid[0], mn[1], mid[2], mx[0], mid[1], mx[2]);
	cBox[6] = Box(mid[0], mid[1], mn[2], mx[0], mx[1], mid[2]);
	cBox[7] = Box(mid[0], mid[1], mid[2], mx[0], mx[1], mx[2]);

	//建立子节点-O(n)
	for (int childID = 0; childID < 8; childID++) {
		std::vector<int> childTrigs;
		for (unsigned int vi = 0; vi < trigs.size(); vi++) {
			int trigIdx = trigs[vi];
			//Box tBox = trigBox(trigIdx, m);
			if (boxOverlap(&(m.getTriangles()[trigIdx].AABB), &(cBox[childID])))
				childTrigs.push_back(trigIdx);
		}
		buildNode(parent->child[childID], cBox[childID], childTrigs, m, level);
	}
}

void Octree::build(Mesh* m)
{
	mesh = m;

	const auto& tri = mesh->getTriangles();
	assert(!tri.empty());

	//计算包含所有面片的包围盒-O(n)
	box.Min = tri[0].getVertex(0);
	box.Max = tri[0].getVertex(0);
	for (unsigned int triID = 0; triID < tri.size(); triID++) {
		const auto& t = tri[triID];
		for (int vi = 0; vi < 3; ++vi) {
			const auto& v = t.getVertex(vi);
			for (int dim = 0; dim < 3; dim++) {
				if (box.Min[dim] > v[dim]) {
					box.Min[dim] = v[dim];
				}
				if (box.Max[dim] < v[dim]) {
					box.Max[dim] = v[dim];
				}
			}
		}
	}

	//转换为三角标号
	std::vector<int> trigs(tri.size());
	for (unsigned int triID = 0; triID < trigs.size(); triID++) {
		trigs[triID] = triID;
	}

	buildNode(&root, box, trigs, *m, 0);
}

int first_node(float tx0, float ty0, float tz0,float txm, float tym, float tzm)
{
	int bits = 0;
	///find max x0 y0 z0
	if (tx0 > ty0) {
		if (tx0 > tz0) { // PLANE YZ
			if (tym < tx0) {
				bits |= 2;
			}
			if (tzm < tx0) {
				bits |= 1;
			}
			return bits;
		}
	}
	else {
		if (ty0 > tz0) {
			if (txm < ty0) {
				bits |= 4;
			}
			if (tzm < ty0) {
				bits |= 1;
			}
			return bits;
		}
	}
	if (txm < tz0) {
		bits |= 4;
	}
	if (tym < tz0) {
		bits |= 2;
	}
	return bits;
}

//以三个方向中最小者作为下一个递归的子节点
int new_node(float txm, int x, float tym, int y, float tzm, int z)
{
	if (txm < tym) {
		if (txm < tzm) {
			return x;
		}
	}
	else {
		if (tym < tzm) {
			return y;
		}
	}
	return z;
}

bool Octree::proc_subtree(float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, OctNode* node,
						const Ray& ray, Hit& h, float tmin, int& objIndex, IntersectMode mode)
{
	bool intersected = false;

	if (tx1 < 0 || ty1 < 0 || tz1 < 0) {
		return intersected;
	}

	if (node->Leaf()) {
		for (size_t ii = 0; ii < node->obj.size(); ii++) {
			bool result = mesh->interID(node->obj[ii], ray, h, tmin, objIndex, mode);
			intersected = intersected || result;
		}
		return intersected;
	}

	float txm = 0.5f * (tx0 + tx1);
	float tym = 0.5f * (ty0 + ty1);
	float tzm = 0.5f * (tz0 + tz1);
	int currNode = first_node(tx0, ty0, tz0, txm, tym, tzm);
	do {
		switch (currNode) {
		case 0: {
			bool result = proc_subtree(tx0, ty0, tz0, txm, tym, tzm, node->child[aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(txm, 4, tym, 2, tzm, 1);
		} break;
		case 1: {
			bool result = proc_subtree(tx0, ty0, tzm, txm, tym, tz1, node->child[1 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(txm, 5, tym, 3, tz1, 8);
		} break;
		case 2: {
			bool result = proc_subtree(tx0, tym, tz0, txm, ty1, tzm, node->child[2 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(txm, 6, ty1, 8, tzm, 3);
		} break;
		case 3: {
			bool result = proc_subtree(tx0, tym, tzm, txm, ty1, tz1, node->child[3 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(txm, 7, ty1, 8, tz1, 8);
		} break;
		case 4: {
			bool result = proc_subtree(txm, ty0, tz0, tx1, tym, tzm, node->child[4 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(tx1, 8, tym, 6, tzm, 5);
		} break;
		case 5: {
			bool result = proc_subtree(txm, ty0, tzm, tx1, tym, tz1, node->child[5 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(tx1, 8, tym, 7, tz1, 8);
		} break;
		case 6: {
			bool result = proc_subtree(txm, tym, tz0, tx1, ty1, tzm, node->child[6 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = new_node(tx1, 8, ty1, 8, tzm, 7);
		} break;
		case 7: {
			bool result = proc_subtree(txm, tym, tzm, tx1, ty1, tz1, node->child[7 ^ aa], ray, h, tmin, objIndex, mode);
			intersected |= result;
			currNode = 8;
		} break;
		}
	} while (currNode < 8);

	return intersected;
}

bool Octree::intersect(const Ray& ray, Hit& h, float tmin, int& objIndex, IntersectMode mode)
{
	Vector3f dir = ray.getDirection();

	dir = dir.normalized();
	Vector3f org = ray.getOrigin();

	aa = 0;
	Vector3f size = box.Max + box.Min; //包围盒两端点的中点
	//归一化光线方向，便于求交测试
	if (dir[0] < 0.0f) {
		org[0] = size[0] - org[0];
		dir[0] = -dir[0];
		aa |= 4;
	}
	if (dir[1] < 0.0f) {
		org[1] = size[1] - org[1];
		dir[1] = -dir[1];
		aa |= 2;
	}
	if (dir[2] < 0.0f) {
		org[2] = size[2] - org[2];
		dir[2] = -dir[2];
		aa |= 1;
	}

	float divx = 1 / dir[0]; // IEEE stability fix
	float divy = 1 / dir[1];
	float divz = 1 / dir[2];

	//t0:最近交点;t1:最远交点
	float tx0 = (box.Min[0] - org[0]) * divx;
	float tx1 = (box.Max[0] - org[0]) * divx;
	float ty0 = (box.Min[1] - org[1]) * divy;
	float ty1 = (box.Max[1] - org[1]) * divy;
	float tz0 = (box.Min[2] - org[2]) * divz;
	float tz1 = (box.Max[2] - org[2]) * divz;

	//相交则下传
	if (std::max(std::max(tx0, ty0), tz0) <= std::min(std::min(tx1, ty1), tz1)) {
		return proc_subtree(tx0, ty0, tz0, tx1, ty1, tz1, &root, ray, h, tmin, objIndex, mode);
	}
	else {
		return false;
	}
}