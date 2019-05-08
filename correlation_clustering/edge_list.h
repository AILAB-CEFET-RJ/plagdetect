#ifndef EDGE_LIST_H_
#define EDGE_LIST_H_
#include <vector>
#include <algorithm>

#include "edge.h"

using namespace std;

class EdgeList
{
  protected:
	vector<Edge *> graph;

  public:
	typedef vector<Edge *>::iterator iterator;
	typedef vector<Edge *>::const_iterator const_iterator;
	EdgeList();
	void push_back(Edge *const edge)
	{
		graph.push_back(edge);
	}
	/*Sort by edge cost*/
	void sort();

	const Edge *operator[](size_t i) const
	{
		return graph[i];
	}
	Edge *at(size_t i) const
	{
		return graph[i];
	}
	size_t size() { return graph.size(); }
	iterator begin() { return graph.begin(); }
	iterator end() { return graph.end(); }
	virtual ~EdgeList();
};

#endif /* EDGE_LIST_H_ */
