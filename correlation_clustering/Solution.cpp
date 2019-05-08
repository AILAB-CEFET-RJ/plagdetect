#ifndef SOLUTION_CPP_
#define SOLUTION_CPP_

#include "Solution.h"

int min_element(vector<pair<int, vector<int>>> ::iterator vec) {
	int min;
	for(auto it = vec->second.begin(); it!= vec->second.end();++it) {
		if(it==vec->second.begin()) min = *it;
		else if(min>*it) min = *it;
	}
	return min;
}

Solution::Solution(Instance* _instance) :
		instance(_instance) {
	cost = 10000000;
}

Solution::~Solution() {
	delete instance;
}

void Solution::computeCost() {

}

void Solution::setCost(double i) {
	cost = i;
}

void Solution::addCluster(int i) {
	vector<int> cluster;
	cluster.push_back(i);
	clusters.push_back(make_pair(i, cluster));
}

void Solution::addNode(int i, int rep) {
	bool found = false;
	vector<pair<int, vector<int>>> ::iterator c = getCluster(rep);
	if (i < rep)
		c->first = i;
	c->second.push_back(i);
}

vector<pair<int, vector<int>>> ::iterator Solution::getCluster(int rep) {
	for(auto it = clusters.begin(); it!=clusters.end();++it) {
		if(it->first == rep) return it;
	}
	cerr<<"Cluster com rep "<<rep<<" não foi encontrado"<<endl;
	exit(1);
}

// annealling movimento de perturbacao pegar dois caras e trocar, pegar quatro caras e trocar, pegar um elemnto e criar cluster
void Solution::moveNode(int i, int rep, int rep2) {
	vector<pair<int, vector<int>>> ::iterator c1 = getCluster(rep);
	vector<pair<int,vector<int>>>::iterator c2 = getCluster(rep2);

	c1->second.erase(std::remove(c1->second.begin(), c1->second.end(), i), c1->second.end());
	if(i==rep) {
		c1->first = min_element(c1);
	}
	c2->second.push_back(i);
	if(i<rep2) c2->first = i;
}

ostream& operator<<(ostream& strm, const Solution& s) {

	for (int i = 0; i < s.clusters.size(); ++i) {
		strm << "(";
		for (auto it = s.clusters[i].second.begin(); it != s.clusters[i].second.end(); ++it) {
			strm << *it << ",";
		}
		strm << ")" << endl;
	}

	return strm;
}

// Numero de clusuter diferentes / pra cada representante verifica se existe um cluster do msm representante, se tiver verifica se tamanhao eh igual true
// pra cada conjunto msm representante veririficar a lista
bool operator==(Solution &a, Solution &b) {
	if (a.clusters.size() < b.clusters.size() || a.clusters.size() > b.clusters.size())
		return false;

	return false;
}

bool operator<(Solution &a, Solution &b) {

	return a.clusters.size() < b.clusters.size();
}

bool operator>(Solution &a, Solution &b) {

	return a.clusters.size() > b.clusters.size();
}

#endif // SOLUTION_CPP_it2 = p.Neighbor[*it].begin()
