#ifndef MYCALLBACK_CPP_
#define MYCALLBACK_CPP_

#include "mycallback.h"

MyCallback::MyCallback(Instance* _p, LSExpr2D* x)
{
		p = _p;
        lastBestRunningTime = 0;
        lastBestValue = 0;
        this->x = x;
}

void MyCallback::callback(LocalSolver& ls, LSCallbackType type) {
	LSStatistics stats = ls.getStatistics();
	LSExpression obj = ls.getModel().getObjective(0);
	//cout<<"Entrou no Callback"<<endl;

	bool melhorou = false;

	if(obj.getDoubleValue() < lastBestValue) {
		lastBestRunningTime = stats.getRunningTime();
		lastBestValue = obj.getDoubleValue();
	}

	//cout<<"Aqui: "<<ls.getStatistics().getPercentImprovingMoves()<<endl;
	if(stats.getRunningTime() - lastBestRunningTime > 100) {
		cout << ">>>>>>> No improvement during 10 seconds: resolution is stopped" << endl;
    cout << ">>>>>>> Best Solution: "<< lastBestValue << endl;
		ls.stop();
	}
}

#endif//MYCALLBACK_CPP_
