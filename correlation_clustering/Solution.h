#ifndef SOLUTION_H_
#define SOLUTION_H_

// C++
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

// CCP
#include "instance.h"

// Namespace
using namespace std;

class Solution {
public:
	// Constructor
	Solution(Instance* _instance);
	// Destructor
	~Solution();

	///// OPERATOR
	// Less than operator
	friend bool operator<(Solution &a, Solution &b);
	// Greater than operator
	friend bool operator>(Solution &a, Solution &b);
	// Equal operator
	friend bool operator==(Solution &a, Solution &b);
	// Assign operator
	Solution& operator=(const Solution& rhs);
	// Subscript operator
	bool operator()(const Solution& a, const Solution& b) const {
		return (a.cost < b.cost);
	}
	// Print operator
	friend ostream& operator<<(ostream&, const Solution&);

	// GETTER
	double getCost() const {
		return cost;
	}
	int getNumberOfClusuters() const {
		return clusters.size();
	}
	int getClusterId(int i) {
		return clusters[i].first;
	}
	vector<pair<int, vector<int>>> ::iterator getCluster(int representative);

	///// SETTER
	void setCost(double i);

	///// ESSENTIAL
	// Add new cluster
	void addCluster(int i);
	// Add node
	void addNode(int i, int representative);
	// Move node
	void moveNode(int i, int representative, int representative2);
	// Compute cost
	void computeCost();

private:
	// Instance for problem
	Instance* instance;
	// Clusters (representative + other nodes)
	vector<pair<int, vector<int>>> clusters;
	// Solution cost
	double cost;
};

#endif //SOLUTION_H_
