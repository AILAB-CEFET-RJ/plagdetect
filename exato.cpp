#ifndef EXATO_CPP
#define EXATO_CPP

#include "exato.h"
#include <fstream>
#include <iostream>

using std::fstream;
using std::ifstream;
using std::ios;
using std::ofstream;

void Exato::addFO(IloEnv &env, IloModel &modelo, IloNumVarMatrix &x)
{
	IloExpr FO(env);
	for (auto it = instance->begin_edges(); it != instance->end_edges(); ++it)
	{
		Edge *edge = *it;
		if (edge->get_cost() < 0)
		{
			FO += -edge->get_cost() * (1 - x[edge->get_i()][edge->get_j()]);
		}
		else if (edge->get_cost() > 0)
		{
			FO += edge->get_cost() * x[edge->get_i()][edge->get_j()];
		}
		//FO+=(edge->get_cost()<0)?edge->get_cost()*(1-x[edge->get_i()][edge->get_j()]):edge->get_cost()*x[edge->get_i()][edge->get_j()];
	}
	modelo.add(IloMinimize(env, FO));
}

void Exato::addConstraint_together(IloEnv &env, IloModel &modelo, IloNumVarMatrix &x)
{
	for (unsigned int i = 0; i < instance->get_n_nodes(); ++i)
	{
		for (unsigned int j = 0; j < instance->get_n_nodes(); ++j)
		{
			for (unsigned int k = 0; k < instance->get_n_nodes(); ++k)
			{
				if (i != j && i != k && j != k)
				{
					IloRange constr_together(env, 0, x[i][k] + x[k][j] - x[i][j], IloInfinity);
					stringstream name;
					name << "together[" << i << "][" << j << "][" << k << "]: ";
					constr_together.setName(name.str().c_str());
					modelo.add(constr_together);
					constraints.add(constr_together);
				}
			}
		}
	}
}

void Exato::addConstraint_same(IloEnv &env, IloModel &modelo, IloNumVarMatrix &x)
{
	for (unsigned int i = 0; i < instance->get_n_nodes(); ++i)
	{
		for (unsigned int j = 0; j < instance->get_n_nodes(); ++j)
		{
			if (i != j)
			{
				IloRange constr_same(env, 0, x[j][i] - x[i][j], 0);
				stringstream name;
				name << "together[" << i << "][" << j << "]: ";
				constr_same.setName(name.str().c_str());
				modelo.add(constr_same);
				constraints.add(constr_same);
			}
		}
	}
}

Exato::Exato(Instance *_instance)
{
	// Inicia o ambiente Concert

	instance = _instance;
	modelo = IloModel(env);
	constraints = IloRangeArray(env);

	x = IloNumVarMatrix(env, instance->get_n_nodes());

	for (unsigned int i = 0; i < instance->get_n_nodes(); ++i)
	{
		x[i] = IloNumVarArray(env, instance->get_n_nodes(), 0, 1, ILOBOOL);
		for (unsigned int j = 0; j < instance->get_n_nodes(); ++j)
		{
			if (i != j)
			{
				stringstream varx;
				varx << "x[" << i << "][" << j << "]";
				x[i][j].setName(varx.str().c_str());
				modelo.add(x[i][j]);
			}
		}
	}

	addConstraint_same(env, modelo, x);
	addConstraint_together(env, modelo, x);
	addFO(env, modelo, x);
}

Exato::~Exato()
{
	cout << "Deleting Exato ... ";
	CCP.end();
	env.end();
	cout << "done." << endl;
}

void Exato::Branch_and_Bound(int timelimit)
{
	//Solution * sol = new Solution (p);
	CCP = IloCplex(modelo);
	CCP.exportModel("lp-linear.lp");
	CCP.setParam(IloCplex::TiLim, timelimit);

	//Resolver direto pelo Concert
	//CCP.setParam(IloCplex::PreLinear, 0);
	//CCP.setParam(IloCplex::AdvInd, 1); //AdvInd = 1 ou 2
	/*
	CCP.setParam(IloCplex::PreLinear, 0);
	CCP.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
	CCP.setParam(IloCplex::Param::Preprocessing::Presolve, 0);

	CCP.setParam(IloCplex::AdvInd, 0); //AdvInd = 1 ou 2

	CCP.setParam(IloCplex::Threads, 1);

	CCP.setParam(IloCplex::PreInd, IloFalse);


	CCP.setParam(IloCplex::HeurFreq, -1);
	CCP.setParam(IloCplex::Cliques, -1);
	CCP.setParam(IloCplex::Covers, -1);
	CCP.setParam(IloCplex::DisjCuts, -1);
	CCP.setParam(IloCplex::FlowCovers, -1);
	CCP.setParam(IloCplex::FlowPaths, -1);
	CCP.setParam(IloCplex::FracCuts, -1);
	CCP.setParam(IloCplex::GUBCovers, -1);
	CCP.setParam(IloCplex::ImplBd, -1);
	CCP.setParam(IloCplex::MIRCuts, -1);
	CCP.setParam(IloCplex::ZeroHalfCuts, -1);
	CCP.setParam(IloCplex::MCFCuts, -1);
	CCP.setParam(IloCplex::Param::MIP::Cuts::LiftProj, -1);
/*
//#ifdef PRINT
	CCP.setParam(IloCplex::NetDisplay, 0);
	CCP.setParam(IloCplex::SiftDisplay, 0);
	CCP.setParam(IloCplex::SimDisplay, 0);
	CCP.setParam(IloCplex::BarDisplay, 0);
	CCP.setParam(IloCplex::MIPDisplay, 0);
//#endif*/

	CCP.solve();

	if (getStatus() == 0)
	{
		cout << "Optimum Found" << endl;
		cout << "BestSol: " << CCP.getObjValue() << endl;
		cout << "GAP: " << CCP.getMIPRelativeGap() << endl;
		PrintCusters();
	}
	else if (getStatus() == 1)
	{
		cout << "Solution Found" << endl;
		cout << "BestSol: " << CCP.getObjValue() << endl;
		cout << "GAP: " << CCP.getMIPRelativeGap() << endl;
		PrintCusters();
	}
	else if (getStatus() == 2)
	{
		cout << "Infeasible" << endl;
	}
	else
	{
		cout << "Unknown" << endl;
	}

	//return sol;
}

status Exato::getStatus()
{
	if (CCP.getStatus() == IloAlgorithm::Infeasible)
	{
		return INFEASIBLE;
	}
	else if (CCP.getStatus() == IloAlgorithm::Optimal)
	{
		return OPTIMALFOUND;
	}
	else
	{
		return SOLUTIONFOUND;
	}
}

unsigned Exato::getNumConstraints(IloModel m)
{
	unsigned count = 0;
	IloModel::Iterator iter(m);
	while (iter.ok())
	{
		if ((*iter).asConstraint().getImpl())
		{
			++count;
		}
		++iter;
	}
	return count;
}

void Exato::PrintCusters()
{
	vector<set<int>> s;
	int *flag = new int[instance->get_n_nodes()];
	for (unsigned int i = 0; i < instance->get_n_nodes(); ++i)
	{
		flag[i] = -1;
	}

	for (unsigned int i = 0; i < instance->get_n_nodes() - 1; ++i)
	{
		if (flag[i] == -1)
		{
			set<int> cluster;
			cluster.insert(i);
			flag[i] = i;
			for (unsigned int j = i + 1; j < instance->get_n_nodes(); ++j)
			{
				if (CCP.getValue(x[i][j]) == 0)
				{
					cluster.insert(j);
					flag[j] = i;
				}
			}
			s.push_back(cluster);
		}
	}
	for (int i = 0; i < s.size(); ++i)
	{
		cout << "Cluster " << i << ": ";
		for (set<int>::iterator it = s[i].begin(); it != s[i].end(); ++it)
		{
			cout << *it << " ";
		}
		cout << endl;
	}

	delete[] flag;
}

#endif /*EXATO_CPP*/
