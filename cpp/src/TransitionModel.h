#ifndef TRANSITIONMODEL_H
#define TRANSITIONMODEL_H

#include <GClasses/GRand.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GVec.h>
#include <GClasses/GDom.h>
#include "Mentor.h"

using namespace GClasses;


/// A model that maps from current beliefs and actions to anticipated beliefs.
/// This model is trained in a supervised manner.
class TransitionModel {
public:
	GRand& rand;
	GNeuralNet model;
	GMatrix trainInput;
	GMatrix trainOutput;
	Tutor* tutor;
	size_t trainPos;
	size_t trainSize;
	size_t trainIters;
	size_t trainProgress;
	double err;
	double prevErr;
	GVec buf;


	/// General-purpose constructor
	TransitionModel(size_t input_dims, size_t output_dims, size_t total_layers, size_t queue_size, size_t trainItersPerPattern, GRand& r);

	/// Unmarshaling constructor
	TransitionModel(GDomNode* pNode, GRand& r);

	/// Marshals this model to a JSON DOM.
	GDomNode* marshal(GDom* pDoc);

	/// Returns the number of action dims
	size_t actionDims();

	/// Sets the tutor
	void setTutor(Tutor* t) { tutor = t; }

	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining();

	/// Refines this model based on a recently performed action and change in beliefs
	void trainIncremental(const GVec& beliefs, const GVec& actions, const GVec& nextBeliefs);

	/// Predict the belief vector that will result if the specified action is performed
	void anticipateNextBeliefs(const GVec& beliefs, const GVec& actions, GVec& anticipatedBeliefs);

	/// Compute the anticipated belief vector that will result if the specified plan is executed.
	void getFinalBeliefs(const GVec& beliefs, const GMatrix& plan, GVec& outFinalBeliefs);
};

#endif
