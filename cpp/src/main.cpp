/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GRand.h>
#include <GClasses/GMatrix.h>
#include "Agent.h"
#include "Test.h"
#include <vector>
#include "AgentRandy.h"
#include "AgentManic.h"
#include "DriftingPlatform.h"


using namespace GClasses;
using std::cerr;
using std::cout;
using std::vector;


void gauntlet(std::vector<Agent*>& agents, std::vector<Test*>& tests)
{
	GMatrix results(tests.size(), agents.size());

	// Evaluate every agent against every test
	for(size_t i = 0; i < tests.size(); i++)
	{
		Test& challenge = *tests[i];
		for(size_t j = 0; j < agents.size(); j++)
		{
			Agent& agent = *agents[j];
			double result = challenge.test(agent);
			results.row(i)[j] = result;
		}
	}

	cout << "\n\n";
	cout << "-------------\n";
	cout << "Final results\n";
	cout << "-------------\n";
	cout << "[" << agents[0]->getName();
	for(size_t i = 1; i < agents.size(); i++) {
		cout << "," << agents[i]->getName();
	}
	cout << "]\n";
	results.print(cout);
}

void doit()
{

	GRand r(1234);
	
	// Make a list of agents
	vector<Agent*> agents;
	agents.push_back(new AgentRandy(r));
	agents.push_back(new AgentManic(r));

	// Make a list of tests
	vector<Test*> tests;
	tests.push_back(new DriftingPlatform(r));

	// Run the agents through the gauntlet
	gauntlet(agents, tests);

	for(size_t i = 0; i < agents.size(); i++)
		delete(agents[i]);
	for(size_t i = 0; i < tests.size(); i++)
		delete(tests[i]);
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 0;
	try
	{
		GArgReader args(argc, argv);
		doit();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

