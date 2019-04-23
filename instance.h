#ifndef INSTANCE_H
#define INSTANCE_H
#include <iostream>
#include <fstream>
#include <exception>
#include <utility>
#include <vector>
#include "edge.h"
#include "edge_list.h"

using namespace std;

class Instance
{
  private:
	int n_nodes;
	int n_edges;
	double var_erro;

	EdgeList edges;
	EdgeList *edges_neighbor;

  public:
	int get_n_nodes() { return n_nodes; };
	int get_n_edges() { return n_edges; };
	double get_var_erro() { return var_erro; };

	void load_file(const char *const file_name);

	EdgeList::iterator begin_edges() { return edges.begin(); }
	EdgeList::iterator end_edges() { return edges.end(); }
	EdgeList::iterator begin_edges_neighbor(int i) { return edges_neighbor[i].begin(); }
	EdgeList::iterator end_edges_neighbor(int i) { return edges_neighbor[i].end(); }

	size_t get_size_neighbor(int i) { return edges_neighbor[i].size(); }

	void PrintInstance();
	virtual ~Instance();
};

#endif
