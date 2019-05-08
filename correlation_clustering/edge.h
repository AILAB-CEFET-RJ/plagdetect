#ifndef EDGE_H_
#define EDGE_H_
#include <iostream>
#include <algorithm>

class Edge
{
  private:
	int i, j;
	/*Edge id*/
	int id;
	/*The cost for choose this edge to be in the solution*/
	double cost;

	void init(int _i, int _j, int _id, double _cost);
	friend std::ostream &operator<<(std::ostream &, const Edge &);

  public:
	Edge(int i, int j, int id, double cost);

	virtual ~Edge();
	int get_i() { return i; }
	int get_j() { return j; }
	int get_id() { return id; }
	double get_cost() { return cost; }

	bool operator<(const Edge &e) const;
	bool operator>(const Edge &e) const;
	bool operator==(const Edge &e) const;
	size_t hash() const;
};

#endif /* EDGE_H_ */
