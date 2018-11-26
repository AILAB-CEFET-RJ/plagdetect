
#ifndef HLS_CPP_
#define HLS_CPP_

#include "HLS.h"
void HLS::addConstraint_together(LSModel& model)
{
	for (unsigned int i = 0; i < p->getNumberOfNodes(); ++i){
		for(unsigned int j = 0; j < p->getNumberOfNodes(); ++j)
		{
			for(unsigned int k = 0; k < p->getNumberOfNodes(); ++k)
			{
				if(i!=j && i!=k && j!=k){
					model.constraint(x[i][k] + x[k][j] - x[i][j]<=0);
				}
			}
		}
	}
}

void HLS::addConstraint_same(LSModel& model){
	for (unsigned int i = 0; i < p->getNumberOfNodes(); ++i)
	{
		for(unsigned int j = 0; j < p->getNumberOfNodes(); ++j)
		{
			if(i!=j){
				model.constraint(x[j][i] == x[i][j]);
			}
		}
	}
}

void HLS::addFO(LSModel& model){
	LSExpression FO = model.sum();
	for(auto it = p->beginEdges(); it!= p->endEdges(); ++it){
		Edge * edge = *it;
		if(edge->getCost()<0){
			FO += -edge->getCost()*(1-x[edge->getI()][edge->getJ()]);
		}else if(edge->getCost()>0){
			FO += edge->getCost()*x[edge->getI()][edge->getJ()];
		}
		//FO+=(edge->get_cost()<0)?edge->get_cost()*(1-x[edge->get_i()][edge->get_j()]):edge->get_cost()*x[edge->get_i()][edge->get_j()];
	}

	model.minimize(FO);
}



HLS::HLS(Instance* _p)
{
	p = _p;
	model = localsolver.getModel();

	// Decision variables x[i]
	x.resize(p->getNumberOfNodes());
	for (int i = 0; i < p->getNumberOfNodes(); ++i) {
	        x[i].resize(p->getNumberOfNodes());
		for(int j=0; j< p->getNumberOfNodes(); ++j){
			if(i!=j){
				x[i][j] = model.boolVar();
			}
		}
    }

	cerr<<"Adicionando Restrição - Together"<<endl;
	addConstraint_together(model);
	cerr<<"Adicionando Restrição - Same"<<endl;
	addConstraint_same(model);
	cerr<<"Adicionando FO"<<endl;
	addFO(model);

	model.close();
}


void HLS::solve(int t_heur){
   try {


	   /* Parameterizes the solver. */
	   LSPhase phase = localsolver.createPhase();
	   phase.setTimeLimit(t_heur);

	   LSParam P = localsolver.getParam();
	   //P.setVerbosity(0);
	   P.setNbThreads(4);


	  // MyCallback cb(p, &x);
	  // localsolver.addCallback(CT_TimeTicked, &cb);
	   localsolver.solve();


   } catch (const LSException& e) {
	   cout << "LSException:" << e.getMessage() << endl;
	   exit(1);
   }
}

HLS::~HLS()
{

}

#endif //HLS_CPP_
