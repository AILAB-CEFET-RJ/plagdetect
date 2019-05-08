#ifndef MYCALLBACK_H_
#define MYCALLBACK_H_

#include "localsolver.h"
#include "lscallback.h"
#include <iostream>
#include "Instance.h"

using namespace localsolver;
using namespace std;

typedef vector < LSExpression> LSExpr1D;
typedef vector < LSExpr1D > LSExpr2D;
typedef vector < LSExpr2D > LSExpr3D;
typedef vector < LSExpr3D > LSExpr4D;

class MyCallback : public LSCallback{

private:
    int lastBestRunningTime;
    double lastBestValue;
    Instance* p;
    LSExpr2D *x;

public:
    MyCallback(Instance* _p, LSExpr2D* x);
    void callback(LocalSolver& ls, LSCallbackType type);

};


#endif//MYCALLBACK_H_
