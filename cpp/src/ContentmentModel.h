#ifndef CONTENTMENTMODEL_H
#define CONTENTMENTMODEL_H

#include <GClasses/GMatrix.h>
#include <GClasses/GDom.h>
#include <GClasses/GRand.h>
#include <GClasses/GNeuralNet.h>
#include "Mentor.h"

using namespace GClasses;


/// A model that maps from anticipated beliefs to contentment (or utility).
/// This model is trained by reinforcement from a mentor.
class ContentmentModel
{
public:
	GRand& rand;
	GNeuralNet model;
	GMatrix samples;
	GMatrix contentment;
	Tutor* tutor;
	size_t trainPos;
	size_t trainSize;
	size_t trainIters;
	double learningRate;
	size_t trainProgress;
	double err;
	GVec targBuf;


	// General-purpose constructor
	ContentmentModel(size_t beliefDims, size_t total_layers, size_t queue_size, size_t trainItersPerPattern, GRand& r);

	/// Unmarshaling constructor
	ContentmentModel(GDomNode* pNode, GRand& r);

	/// Marshals this model to a JSON DOM.
	GDomNode* marshal(GDom* pDoc);

	/// Sets the tutor
	void setTutor(Tutor* t) { tutor = t; }

	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining();

	/// Refines this model based on feedback from the mentor
	void trainIncremental(const GVec& sample_beliefs, double sample_contentment);

	/// Computes the contentment of a particular belief vector
	double evaluate(const GVec& beliefs);
};


#endif
