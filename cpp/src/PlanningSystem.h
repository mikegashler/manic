#ifndef PLANNINGSYSTEM_H
#define PLANNINGSYSTEM_H

#include <GClasses/GRand.h>
#include <GClasses/GVec.h>
#include <GClasses/GDom.h>
#include <GClasses/GMatrix.h>
#include <iostream>
#include "Mentor.h"
#include "TransitionModel.h"
#include "ObservationModel.h"
#include "ContentmentModel.h"

using namespace GClasses;


/// A genetic algorithm that sequences actions to form a plan intended to maximize contentment.
class PlanningSystem
{
public:
	Agent& self;
	std::vector<GMatrix*> plans;
	TransitionModel& transitionModel;
	ObservationModel& observationModel;
	ContentmentModel& contentmentModel;
	Mentor* mentor;
	Tutor* tutor;
	size_t maxPlanLength;
	size_t refinementIters;
	size_t actionDims;
	size_t burnIn;
	double discountFactor;
	double explorationRate;
	GRand& rand;
	GMatrix randomPlan;
	GVec buf;
	GVec buf2;


	// General-purpose constructor
	PlanningSystem(Agent& agent, TransitionModel& transition, ObservationModel& observation, ContentmentModel& contentment, Mentor* oracle,
		size_t actionDimensions, size_t populationSize, size_t planRefinementIters, size_t burnInIters, size_t maxPlanLen, double discount, double explore, GRand& r);

	/// Unmarshaling constructor
	PlanningSystem(GDomNode* pNode, Agent& agent, GRand& r, TransitionModel& transition, ObservationModel& observation, ContentmentModel& contentment, Mentor* oracle);

	~PlanningSystem();

	/// Marshals this model to a JSON DOM.
	GDomNode* marshal(GDom* pDoc);

	/// Replaces the mentor with the specified one
	void setMentor(Mentor* oracle);

	/// Sets the tutor
	void setTutor(Tutor* t) { tutor = t; }

	/// Prints a representation of all the plans to stdout
	void printPlans();

	/// Perturbs a random plan
	void mutate();

	/// Replaces the specified plan with a new one.
	void replace(size_t childIndex);

	/// Returns the expected contentment at the end of the plan
	double evaluatePlan(const GVec& beliefs, GMatrix& plan);

	/// Performs a tournament between two randomly-selected plans.
	/// One of them, usually the winner, is replaced.
	void tournament(const GVec& beliefs);

	/// Performs several iterations of plan refinement
	void refinePlans(const GVec& beliefs);

	/// Drops the first action in every plan
	void advanceTime();

	/// Asks the mentor to evaluate the plan, given our current beliefs, and learn from it
	void askMentorToEvaluatePlan(const GVec& beliefs, GMatrix& plan);

	/// Finds the best plan and copies its first step
	void chooseNextActions(const GVec& beliefs, GVec& actions);
};

#endif
