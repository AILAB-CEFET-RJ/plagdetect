#include "edge_list.h"

EdgeList::EdgeList()
{
	// TODO Auto-generated constructor stub
}

typedef struct _edge_list_item_cmp
{
	bool operator()(Edge *const a, Edge *const b) const
	{
		if (a->get_i() < b->get_i())
			return true;
		else if (a->get_i() > b->get_i())
			return false;
		else
			return *a < *b;
	}
} edge_list_item_cmp;

void EdgeList::sort()
{
	std::sort(graph.begin(), graph.end(), edge_list_item_cmp());
}

EdgeList::~EdgeList()
{
	// TODO Auto-generated destructor stub
}
