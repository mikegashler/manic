#ifndef DRIFTINGPLATFORM_H
#define DRIFTINGPLATFORM_H

#include "Mentor.h"
#include "Agent.h"
#include "Test.h"
#include <GClasses/GVec.h>
#include <GClasses/GRand.h>
#include <GClasses/GMatrix.h>


class DriftingPlatformMentor : public Mentor
{
public:
	bool active;

	DriftingPlatformMentor()
	{
		active = true;
	}

	// Prefer the fantasy that minimizes the magnitude of the observation vector
	virtual double evaluate(const GVec& anticipatedObservations)
	{
		if(!active)
			return UNKNOWN_REAL_VALUE;

		double sqMag = anticipatedObservations.squaredMagnitude();
		return exp(-sqMag);
	}
};



class DriftingPlatformTutor : public Tutor
{
public:
	double controlOrigin;
	double stepSize;
	DriftingPlatformMentor& mentor;
	GClasses::GVec obs;

	DriftingPlatformTutor(DriftingPlatformMentor& m) : controlOrigin(0.0), stepSize(0.05), mentor(m), obs(2)
	{
	}

	virtual void observations_to_state(const GClasses::GVec& observations, GClasses::GVec& state)
	{
		state.put(0, observations, 0, state.size());
	}

	virtual void state_to_observations(const GClasses::GVec& state, GClasses::GVec& observations)
	{
		observations.put(0, state, 0, state.size());
		observations.fill(0.0, state.size(), observations.size());
	}

	virtual void transition(const GClasses::GVec& current_state, const GClasses::GVec& actions, GClasses::GVec& next_state)
	{
		next_state.copy(current_state);
		double angle = actions[0] * 2.0 * M_PI + controlOrigin;
		next_state[0] += stepSize * std::cos(angle);
		next_state[1] += stepSize * std::sin(angle);
		next_state.clip(-1.0, 1.0);
	}

	virtual double evaluate_state(const GClasses::GVec& state)
	{
		state_to_observations(state, obs);
		bool oldActive = mentor.active;
		mentor.active = true;
		double utility = mentor.evaluate(obs);
		mentor.active = oldActive;
		return utility;
	}

	virtual void choose_actions(const GClasses::GVec& state, GClasses::GVec& actions)
	{
		double theta = atan2(state[1], state[0]);
		theta -= controlOrigin;
		theta += M_PI;
		theta /= (2.0 * M_PI);
		while(theta < 0.0)
			theta += 1.0;
		while(theta > 1.0)
			theta -= 1.0;
		actions[0] = theta;
	}
};



class DriftingPlatform : public Test
{
public:
	GRand& rand;


	DriftingPlatform(GRand& r)
	: rand(r)
	{
	}

/*
	/// Generates an image to visualize what's going on inside an AgentManic's artificial brain for debugging purposes
	static BufferedImage visualize(agents.manic.AgentManic agent, double[] state_orig, double[] state_drifted, double[] state);
	static void makeVisualization(String suffix, agents.manic.AgentManic agent, double[] state_orig, double[] state_drifted, double[] state);
*/

	double test(Agent& agent);
};

#endif
