#include "DriftingPlatform.h"
#include <iostream>

//#include "AgentManic.h"
//#include "GClasses/GImage.h"

using std::cout;
/*
void visualizeContentment(AgentManic& agent)
{
	GImage image;
	image.setSize(1000, 1000);
	GVec beliefs(2);
	
	// Find the min and max contentment
	ContentmentModel* pContentmentModel = agent.contentmentModel;
	double min = 1e300;
	double max = -1e300;
	for(int y = 0; y < (int)image.height(); y++)
	{
		beliefs[1] = (double)y / 1000.0 * 2.0 - 1.0;
		for(int x = 0; x < (int)image.width(); x++)
		{
			beliefs[0] = (double)x / 1000.0 * 2.0 - 1.0;
			double contentment = pContentmentModel->evaluate(beliefs);
			min = std::min(min, contentment);
			max = std::max(max, contentment);
		}
	}
	
	// Plot the contentment contours
	for(int y = 0; y < (int)image.height(); y++)
	{
		beliefs[1] = (double)y / 1000.0 * 2.0 - 1.0;
		for(int x = 0; x < (int)image.width(); x++)
		{
			beliefs[0] = (double)x / 1000.0 * 2.0 - 1.0;
			double contentment = pContentmentModel->evaluate(beliefs);
			int g = ClipChan((contentment - min) * 256.0 / (max - min));
			int gg = g;
			if(g % 5 == 0)
				gg = (128 - (int)(tanh((double)(g - 128) * 0.03) * 127.0));
			image.setPixel(x, y, gARGB(0xff, g, g, gg));
		}
	}

	// Draw magenta dots at the sample locations for training the contentment function
	GVec r;
	for(size_t i = 0; i < agent.contentmentModel->trainSize; i++) {
		agent.observationModel->beliefsToObservations(agent.contentmentModel->samples.row(i), r);
		int x = (int)((r[0] + 1.0) * 500.0);
		int y = (int)((r[1] + 1.0) * 500.0);
		image.circleFill(x, y, 4.0, 0xffff00ff);
	}

	// Draw crosshairs at the origin
	image.line(449, 499, 549, 499, 0xff000000);
	image.line(499, 449, 499, 549, 0xff000000);

	string s = "min: ";
	s += to_str(min);
	s += ",    max: ";
	s += to_str(max);
	image.text(s.c_str(), 50, 50, 2.0f, 0xffffff00);
	
	image.savePpm("contentment.ppm");
}
*/
double DriftingPlatform::test(Agent& agent)
{
	cout << "----------------------\n";
	cout << "Drifting platform test	   Agent: " + agent.getName() << "\n";
	cout << "----------------------\n";
	cout << "In this test, the agent is placed on an imaginary 2D platform of infinite size. " <<
			"The agent's objective is to stay near the origin. Each time-step, the platform " <<
			"drifts a small amount in a random direction. The agent can step in any direction "<<
			"(from 0 to 2*PI). Initially, a mentor will help it learn what to do.\n";

	// Define some constants for this test
	double driftSpeed = 0.1;

	// Make an agent
	DriftingPlatformMentor mentor;
	agent.reset(mentor, // This mentor prefers plans that lead closer to the origin
		2, // The agent observes its x,y position (which is the complete state of this world)
		2, // the agent models state with 2 dimensions because it cannot be simplified further
		1, // The agent chooses a direction for travel
		1); // The agent plans up to 10 time-steps into the future
	DriftingPlatformTutor tutor(mentor);

	// To debug an agent that isn't working, uncomment the following line and verify that it works.
	// Then, set each "true" to "false" until you find the component that isn't doing its job properly.
	//agent.setTutor(&tutor, false/*observation*/, false/*transition*/, false/*contentment*/, false/*planning*/);

	// Train with mentor
	cout << "Phase 1 of 3: Supervised learning...\n";
	cout << "|------------------------------------------------|\n";
	GVec state(2);
	GVec obs(2);
	GVec next_state(2);
	GVec drift(2);
	for(size_t i = 0; i < 2000; i++) {

		if(i % 40 == 0)
		{
			cout << ">";
			cout.flush();
		}

		// The platform drifts in a random direction
		drift[0] = rand.normal();
		drift[1] = rand.normal();
		drift.normalize();
		drift *= driftSpeed;
		state += drift;
		state.clip(-1.0, 1.0);

		// The agent takes a step in a direction of its choice
		tutor.state_to_observations(state, obs);
		GVec& act = agent.think(obs);
		tutor.transition(state, act, next_state);
		state.copy(next_state);
	}
	//if(agent.getName().compare("manic") == 0)
	//	visualizeContentment(*(AgentManic*)&agent);

	cout << "\n\n\nNow, the mentor is removed, so the agent is on its own.\n" ;
	mentor.active = false;

	cout << "Also, to make the problem more challenging, the agent's controls " <<
			"are changed by 120 degrees. The agent will now have to figure out how to operate " <<
			"the new controls without a mentor to help it.\n";
	tutor.controlOrigin += M_PI * 2.0 / 3.0;

	// Train without mentor
	cout << "Phase 2 of 3: Unsupervised learning...\n";
	cout << "|------------------------------------------------|\n";
	for(size_t i = 0; i < 2000; i++) {

		if(i % 40 == 0)
		{
			cout << ">";
			cout.flush();
		}

		// The platform drifts in a random direction
		drift[0] = rand.normal();
		drift[1] = rand.normal();
		drift.normalize();
		drift *= driftSpeed;
		state += drift;
		state.clip(-1.0, 1.0);

		// The agent takes a step in a direction of its choice
		tutor.state_to_observations(state, obs);
		GVec& act = agent.think(obs);
		tutor.transition(state, act, next_state);
		state.copy(next_state);
	}

	// Test
	cout << "\n\n\nThe agent has had enough time to figure out the new controls, so now we test the agent. " <<
			"We will let the platform continue to drift randomly for 1000 iterations, and measure the average " <<
			"distance between the origin and the agent. (If the agent is intelligent, it should achieve a low " <<
			"average distance, such as 0.2. If it is unintelligent, it will achieve a higher average distance, " <<
			"such as 0.7.\n";
	cout << "Phase 3 of 3: Testing...\n";
	cout << "|------------------------------------------------|\n";
	double sumSqMag = 0.0;
	for(size_t i = 0; i < 1000; i++) {

		if(i % 20 == 0)
		{
			cout << ">";
			cout.flush();
		}

		// The platform drifts in a random direction
		drift[0] = rand.normal();
		drift[1] = rand.normal();
		drift.normalize();
		drift *= driftSpeed;
		state += drift;
		state.clip(-1.0, 1.0);

		// The agent takes a step in a direction of its choice
		tutor.state_to_observations(state, obs);
		GVec& act = agent.think(obs);
		tutor.transition(state, act, next_state);
		state.copy(next_state);

		// Sum up how far the agent ever drifts from the origin
		sumSqMag += std::sqrt(state.squaredMagnitude());
	}

	double aveDist = sumSqMag / 1000.0;
	cout << "\n\nThe agent's average distance from the origin during the testing phase was " << to_str(aveDist) << "\n\n";

	return -aveDist; // Bigger is supposed to be better, so we negate the average distance
}
