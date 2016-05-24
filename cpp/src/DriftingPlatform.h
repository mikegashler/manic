#ifndef DRIFTINGPLATFORM_H
#define DRIFTINGPLATFORM_H

#include "Mentor.h"
#include "Agent.h"
#include "Test.h"
#include <GClasses/GVec.h>
#include <GClasses/GRand.h>
#include <GClasses/GMatrix.h>


class DriftingPlatform;


class DriftingPlatformMentor : public Mentor
{
public:
	bool active;
	GVec anticipatedObs;

	DriftingPlatformMentor()
	{
		active = true;
	}

	virtual double evaluatePlan(Agent& agent, const GMatrix& plan)
	{
		if(!active)
			return UNKNOWN_REAL_VALUE;
		agent.anticipateObservation(plan, anticipatedObs);
		return evaluateObservation(anticipatedObs);
	}

	// Prefer the fantasy that minimizes the magnitude of the observation vector
	static double evaluateObservation(const GVec& anticipatedObservations)
	{
		double sqMag = anticipatedObservations.squaredMagnitude();
		return exp(-sqMag);
	}
};



class DriftingPlatformTutor : public Tutor
{
public:
	DriftingPlatform& world;
	DriftingPlatformMentor& mentor;
	GClasses::GVec obs;

	DriftingPlatformTutor(DriftingPlatform& w, DriftingPlatformMentor& m) : world(w), mentor(m), obs(2)
	{
	}

	virtual void observations_to_state(const GClasses::GVec& observations, GClasses::GVec& state);
	virtual void state_to_observations(const GClasses::GVec& state, GClasses::GVec& observations);
	virtual void transition(const GClasses::GVec& current_state, const GClasses::GVec& actions, GClasses::GVec& next_state);
	virtual double evaluate_state(const GClasses::GVec& state);
	virtual void choose_actions(const GClasses::GVec& state, GClasses::GVec& actions);
};



class DriftingPlatform : public Test
{
public:
	double controlOrigin;
	double stepSize;
	GRand& rand;


	DriftingPlatform(GRand& r)
	: controlOrigin(0.0), stepSize(0.05), rand(r)
	{
	}

	void computeObservations(const GClasses::GVec& state, GClasses::GVec& observations)
	{
		observations.put(0, state, 0, state.size());
		observations.fill(0.0, state.size(), observations.size());
	}

	void computeNextState(const GClasses::GVec& current_state, const GClasses::GVec& actions, GClasses::GVec& next_state)
	{
		next_state.copy(current_state);
		double angle = actions[0] * 2.0 * M_PI + controlOrigin;
		next_state[0] += stepSize * std::cos(angle);
		next_state[1] += stepSize * std::sin(angle);
		next_state.clip(-1.0, 1.0);
	}


/*
	/// Generates an image to visualize what's going on inside an AgentManic's artificial brain for debugging purposes
	static BufferedImage visualize(agents.manic.AgentManic agent, double[] state_orig, double[] state_drifted, double[] state);
	static void makeVisualization(String suffix, agents.manic.AgentManic agent, double[] state_orig, double[] state_drifted, double[] state);
*/

	double test(Agent& agent);
};

#endif
