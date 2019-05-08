#ifndef EXATO_H
#define EXATO_H

enum status
{
	OPTIMALFOUND,
	SOLUTIONFOUND,
	INFEASIBLE
};

#include <ilcplex/ilocplex.h>

#include <iostream>
#include <utility>
#include <vector>
#include <set>

#include "instance.h"

ILOSTLBEGIN

#define SMALL_NUMBER 0.000001

typedef IloArray<IloNumVarArray> IloNumVarMatrix;
typedef IloArray<IloNumVarMatrix> IloNumVar3Matrix;
typedef IloArray<IloNumVar3Matrix> IloNumVar4Matrix;

typedef IloArray<IloNumArray> IloNumMatrix;
typedef IloArray<IloNumMatrix> IloNum3Matrix;

using namespace std;

class Exato
{
  private:
	Instance *instance;

	IloConstraintArray constraints;

	IloEnv env;
	IloModel modelo;
	IloCplex CCP;

	IloNumVarMatrix x;

	void addFO(IloEnv &env, IloModel &modelo, IloNumVarMatrix &x);
	void addConstraint_together(IloEnv &env, IloModel &modelo, IloNumVarMatrix &x);
	void addConstraint_same(IloEnv &env, IloModel &modelo, IloNumVarMatrix &x);

	unsigned getNumConstraints(IloModel m);
	void PrintCusters();

	public : Exato(Instance *_instance);

	~Exato();

	void Branch_and_Bound(int timelimit);

	status getStatus();
	IloEnv getEnv() { return env; };

	void PrintSol(){};

};

#endif /*EXATO_H*/
