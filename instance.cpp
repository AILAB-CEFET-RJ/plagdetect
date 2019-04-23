#include "instance.h"

void Instance::load_file(const char *const file_name)
{

	ifstream f_inst(file_name);
	if (!f_inst.is_open())
	{
		cout << "ERROR: File " << file_name << " not found!" << endl;
		exit(0);
	}

	n_edges = 0;

	f_inst >> n_nodes;
	f_inst >> n_nodes;
	f_inst >> n_nodes;
	f_inst >> var_erro;

	edges_neighbor = new EdgeList[n_nodes];

	//int i, j, cost; ivair
        int i, j;
        double cost;        

	while (f_inst >> i)
	{
		//f_inst >> i;
		f_inst >> j;
		f_inst >> cost;
		Edge *edge;
		if (i < j)
			edge = new Edge(i - 1, j - 1, n_edges, cost);
		else
			edge = new Edge(j - 1, i - 1, n_edges, cost);

		edges.push_back(edge);

		edges_neighbor[edge->get_i()].push_back(edge);
		edges_neighbor[edge->get_j()].push_back(edge);
		++n_edges;
	}
	edges.sort();
	f_inst.close();

	//PrintInstance();
}

void Instance::PrintInstance()
{
	cout << n_nodes << endl;
	cout << var_erro << endl;
	cout << n_edges << endl;

	for (auto it = begin_edges(); it != end_edges(); ++it)
	{
		Edge *edge = *it;
		cout << edge->get_i() << " " << edge->get_j() << " " << edge->get_cost() << endl;
	}
}

Instance::~Instance()
{

	cout << "Deleting instance...";
	for (EdgeList::iterator it = begin_edges(); it != end_edges(); ++it)
	{
		Edge *edge = *it;
		delete edge;
	}
	delete[] edges_neighbor;
	cout << "Instance deleted. \n";
}
