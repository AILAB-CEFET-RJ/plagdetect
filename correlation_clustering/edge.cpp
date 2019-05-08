
#include "edge.h"

Edge::Edge(int i, int j, int id, double cost)
{
	init(i, j, id, cost);
}

Edge::~Edge()
{
	// TODO Auto-generated destructor stub
}

void Edge::init(int _i, int _j, int _id, double _cost)
{
	i = _i;
	j = _j;
	id = _id;
	cost = _cost;
}

bool Edge::operator==(const Edge &e) const
{
	return (i == e.i && j == e.j) || (i == e.j && j == e.i);
}

size_t Edge::hash() const
{
	size_t hash = 0;
	hash = (i ^ j >> 1) << 1;

	return hash;
}

bool Edge::operator<(const Edge &e) const
{
	int _i = i;
	int _j = j;
	int _ei = e.i;
	int _ej = e.j;
	if (_i < _ei)
		return true;
	else if (_i > _ei)
		return false;
	else
		return _j < _ej;
}
bool Edge::operator>(const Edge &e) const
{
	int _i = i;
	int _j = j;
	int _ei = e.i;
	int _ej = e.j;
	if (_i > _ei)
		return true;
	else if (_i < _ei)
		return false;
	else
		return _j > _ej;
}

std::ostream &operator<<(std::ostream &strm, const Edge &e)
{
	return strm << "(" << e.i << "," << e.j << ") - " << e.cost << "\n";
}
