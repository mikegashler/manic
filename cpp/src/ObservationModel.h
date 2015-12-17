#ifndef OBSERVATION_H
#define OBSERVATION_H

#include "TransitionModel.h"
#include <GClasses/GMatrix.h>
#include <GClasses/GDom.h>
#include <GClasses/GRand.h>
#include <GClasses/GNeuralNet.h>
#include "Mentor.h"

using namespace GClasses;


/// A bidirectional model that maps between beliefs and observations.
/// Mapping from observations to beliefs is done by the encoder.
/// Mapping from beliefs to observations is done by the decoder.
/// These two components are trained together in an unsupervised manner as an autoencoder.
class ObservationModel
{
public:
	GRand& rand;
	GNeuralNet decoder;
	GNeuralNet encoder;
	GNeuralNet decoderExperimental;
	GNeuralNet encoderExperimental;
	GMatrix train;
	GMatrix validation;
	Tutor* tutor;
	TransitionModel& transitionModel;
	size_t trainPos;
	size_t trainSize;
	size_t validationPos;
	size_t validationSize;
	size_t trainIters;
	size_t trainProgress;
	size_t calibrationIters;


	/// General-purpose constructor
	ObservationModel(TransitionModel& transition, size_t observation_dims, size_t belief_dims, size_t decoder_layers,
		size_t encoder_layers, size_t queue_size, size_t trainItersPerPattern, size_t calibrationIterations, GRand& r);

	/// Unmarshaling constructor
	ObservationModel(TransitionModel& transition, GDomNode* pNode, GRand& r);

	/// Marshals this model to a JSON DOM.
	GDomNode* marshal(GDom* pDoc);

	/// Sets the tutor
	void setTutor(Tutor* t) { tutor = t; }

	/// Performs one pattern-presentation of stochastic gradient descent and dynamically tunes the learning rate
	void doSomeTraining();

	/// Refines the encoder and decoder based on the new observation.
	void trainIncremental(const GVec& observation);

	/// Refines the beliefs to correspond with actual observations
	void calibrateBeliefs(GVec& beliefs, const GVec& observations);

	/// Decodes beliefs to predict observations
	void beliefsToObservations(const GVec& beliefs, GVec& observations);

	/// Encodes observations to predict beliefs
	void observationsToBeliefs(const GVec& observations, GVec& beliefs);
};

#endif
