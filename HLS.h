#ifndef HLS_H_
#define HLS_H_

#include "localsolver.h"
#include "lscallback.h"
#include "mycallback.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <assert.h>     /* assert */
#include "Instance.h"

using namespace localsolver;
using namespace std;

typedef vector < LSExpression> LSExpr1D;
typedef vector < LSExpr1D > LSExpr2D;
typedef vector < LSExpr2D > LSExpr3D;
typedef vector < LSExpr3D > LSExpr4D;


class HLS
{
private:

	LocalSolver localsolver;
	Instance* p;
	LSModel model;
  	LSExpr2D x;

	void addConstraint_together(LSModel& model);
	void addConstraint_same(LSModel& model);
	void addFO(LSModel& model);


public:

	HLS(Instance* _p);

	~HLS();

	//Solution getSolution();

	void solve(int t_heur);

	//status getStatus();

};

#endif // HLS_H_
