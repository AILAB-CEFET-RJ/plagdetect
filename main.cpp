#include <iostream>
#include <stdlib.h>

#include "instance.h"
#include "exato.h"

using namespace std;

void usage(char *argv[])
{
	cout << "Usage:" << endl;
	cout << "\t" << argv[0] << " <input_instance_name> <tempo_limite>" << endl;
	cout << "\n\t"
		 << "<input_instance_name>: nome do arquivo de entrada" << endl;
	cout << "\n\t"
		 << "<tempo_limite>: tempo maximo de execucao do algoritmo" << endl;
}

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		usage(argv);
	}
	else
	{
		srand(time(NULL));

		const char *datafile = argv[1];
		int timelimit = atoi(argv[2]);
		Instance *instance = new Instance();
		instance->load_file(datafile);

		Exato *model = new Exato(instance);

		model->Branch_and_Bound(3600);


		delete instance;
		delete model;
	}

	return 0;
}