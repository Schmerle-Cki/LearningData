#ifndef GROUP_H
#define GROUP_H


#include "object3d.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include <iostream>
#include <vector>

class Group : public Object3D {

public:

	Group() {

	}

	explicit Group(int num_objects) {
		objects.resize(num_objects);
	}

	~Group() override {

	}

	bool intersect(const Ray& r, Hit& h, float tmin, int& objIndex, IntersectMode mode = Simple) override {
		bool intersected = false;

		for (int i = 0; i < objects.size(); i++)
		{
			bool res = objects[i]->intersect(r, h, tmin, objIndex, mode);
			if (res)objIndex = i;
			intersected |= res;
		}

		return intersected;
	}

	void addObject(int index, Object3D* obj) {
		objects[index] = obj;
	}

	int getGroupSize() {
		return objects.size();
	}

	const Object3D* obj(int index)
	{
		return objects[index];
	}

private:
	std::vector<Object3D*> objects;
};

#endif

