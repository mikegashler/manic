#include "ObservationModel.h"
#include <GClasses/GActivation.h>


/// General-purpose constructor
ObservationModel::ObservationModel(TransitionModel& transition, size_t observation_dims, size_t belief_dims, size_t decoder_layers,
		size_t encoder_layers, size_t queue_size, size_t trainItersPerPattern, size_t calibrationIterations, GRand& r)
: rand(r),
train(queue_size, observation_dims),
validation(queue_size, observation_dims),
tutor(nullptr),
transitionModel(transition),
trainPos(0),
trainSize(0),
validationPos(0),
validationSize(0),
trainIters(0),
trainProgress(0),
calibrationIters(0)
{

	if(belief_dims > observation_dims)
		throw Ex("observation_dims must be >= belief_dims");

	// Init the encoder
	int hidden = std::max((size_t)30, (observation_dims + belief_dims) / 2);
	encoder.addLayer(new GLayerClassic(observation_dims, hidden, new GActivationBend()));
	encoder.addLayer(new GLayerClassic(hidden, belief_dims, new GActivationBend()));
	GUniformRelation relInEnc(observation_dims);
	GUniformRelation relOutEnc(belief_dims);
	encoder.setLearningRate(0.03);
	encoder.beginIncrementalLearning(relInEnc, relOutEnc);

	// Init the decoder
	decoder.addLayer(new GLayerClassic(belief_dims, hidden, new GActivationBend()));
	decoder.addLayer(new GLayerClassic(hidden, observation_dims, new GActivationBend()));
	GUniformRelation relInDec(belief_dims);
	GUniformRelation relOutDec(observation_dims);
	decoder.setLearningRate(0.03);
	decoder.beginIncrementalLearning(relInDec, relOutDec);

	// Make the experimental nets
	decoderExperimental.copyStructure(&decoder);
	encoderExperimental.copyStructure(&encoder);

	// Init the meta-parameters
	trainIters = trainItersPerPattern;
	calibrationIters = calibrationIterations;
}


/// Unmarshaling constructor
ObservationModel::ObservationModel(TransitionModel& transition, GDomNode* pNode, GRand& r) 
: rand(r),
decoder(pNode->field("decoder")),
encoder(pNode->field("encoder")),
decoderExperimental(pNode->field("decoderExperimental")),
encoderExperimental(pNode->field("encoderExperimental")),
train(pNode->field("train")),
validation(pNode->field("validation")),
tutor(nullptr),
transitionModel(transition),
trainPos(pNode->field("trainPos")->asInt()),
trainSize(pNode->field("trainSize")->asInt()),
validationPos(pNode->field("validationPos")->asInt()),
validationSize(pNode->field("validationSize")->asInt()),
trainIters(pNode->field("trainIters")->asInt()),
trainProgress(pNode->field("trainProgress")->asInt()),
calibrationIters(pNode->field("calibrationIters")->asInt())
{
}


/// Marshals this model to a JSON DOM.
GDomNode* ObservationModel::marshal(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "decoder", decoder.serialize(pDoc));
	pNode->addField(pDoc, "encoder", encoder.serialize(pDoc));
	pNode->addField(pDoc, "decoderExperimental", decoderExperimental.serialize(pDoc));
	pNode->addField(pDoc, "encoderExperimental", encoderExperimental.serialize(pDoc));
	pNode->addField(pDoc, "train", train.serialize(pDoc));
	pNode->addField(pDoc, "validation", validation.serialize(pDoc));
	pNode->addField(pDoc, "trainPos", pDoc->newInt(trainPos));
	pNode->addField(pDoc, "trainSize", pDoc->newInt(trainSize));
	pNode->addField(pDoc, "validationPos", pDoc->newInt(validationPos));
	pNode->addField(pDoc, "validationSize", pDoc->newInt(validationSize));
	pNode->addField(pDoc, "trainIters", pDoc->newInt(trainIters));
	pNode->addField(pDoc, "trainProgress", pDoc->newInt(trainProgress));
	pNode->addField(pDoc, "calibrationIters", pDoc->newInt(calibrationIters));
	return pNode;
}


/// Performs one pattern-presentation of stochastic gradient descent and dynamically tunes the learning rate
void ObservationModel::doSomeTraining()
{
	// Train the decoderExperimental and encoderExperimental together as an autoencoder
	double lambda = decoder.learningRate() * 0.00001;
	decoderExperimental.scaleWeights(1.0 - lambda);
	decoderExperimental.diminishWeights(lambda);
	encoderExperimental.scaleWeights(1.0 - lambda);
	encoderExperimental.diminishWeights(lambda);
	size_t index = rand.next(trainSize);
	GVec& observation = train.row(index);
	encoderExperimental.forwardProp(observation);
	GVec& belief = encoderExperimental.outputLayer().activation();
	decoderExperimental.forwardProp(belief);
	decoderExperimental.backpropagate(observation);
	encoderExperimental.backpropagateFromLayer(&decoderExperimental.layer(0));
	encoderExperimental.descendGradient(observation, encoderExperimental.learningRate(), 0.0);
	decoderExperimental.descendGradient(belief, decoderExperimental.learningRate(), 0.0);

	// Since changing the observation function resets the training data for the transition function,
	// we only want to change our perception when it will lead to big improvements.
	// Here, we test whether our experimental model is significantly better than the one we have been using.
	// If so, then the experimental model becomes the new model.
	trainProgress++;
	if(trainProgress >= train.rows())
	{
		// Measure mean squared error
		trainProgress = 0;
		double err1 = 0.0;
		double err2 = 0.0;
		for(size_t i = 0; i < validationSize; i++)
		{
			GVec& targ = validation.row(i);
			encoder.forwardProp(targ);
			decoder.forwardProp(encoder.outputLayer().activation());
			GVec& pred1 = decoder.outputLayer().activation();
			encoderExperimental.forwardProp(targ);
			decoderExperimental.forwardProp(encoderExperimental.outputLayer().activation());
			GVec& pred2 = decoderExperimental.outputLayer().activation();
			for(size_t j = 0; j < targ.size(); j++)
			{
				err1 += (targ[j] - pred1[j]) * (targ[j] - pred1[j]);
				err2 += (targ[j] - pred2[j]) * (targ[j] - pred2[j]);
			}
		}
		err1 = std::sqrt(err1 / validationSize);
		err2 = std::sqrt(err2 / validationSize);
		if(err2 < 0.85 * err1)
		{
			// Update the observation model and reset the training data for the transition function
			encoder.copyWeights(&encoderExperimental);
			decoder.copyWeights(&decoderExperimental);
			transitionModel.trainPos = 0;
			transitionModel.trainSize = 0;
		}
		else if(err1 < 0.85 * err2)
		{
			// This should really never happen
			encoderExperimental.copyWeights(&encoder);
			decoderExperimental.copyWeights(&decoder);
		}
		//std::cout << "Observation error:" << to_str(err1) << ", " << to_str(err2) << "\n";
	}
}


/// Refines the encoder and decoder based on the new observation.
void ObservationModel::trainIncremental(const GVec& observation)
{
	// Buffer the pattern
	GVec* dest;
	if(validationPos < trainPos) {
		dest = &validation.row(validationPos);
		if(++validationPos >= validation.rows())
			validationPos = 0;
		validationSize = std::max(validationSize, validationPos);
	} else {
		dest = &train.row(trainPos);
		trainPos++;
		trainSize = std::max(trainSize, trainPos);
		if(trainPos >= train.rows())
			trainPos = 0;
	}
	*dest = observation;

	// Train
	size_t iters = std::min(trainIters, trainSize);
	for(size_t i = 0; i < iters; i++)
		doSomeTraining();
}


/// Refines the beliefs to correspond with actual observations
void ObservationModel::calibrateBeliefs(GVec& beliefs, const GVec& observations)
{
	if(tutor)
		tutor->observations_to_state(observations, beliefs);
	else
	{
		GNeuralNetLayer& layIn = encoder.outputLayer();
		for(size_t i = 0; i < calibrationIters; i++) {
			decoder.forwardProp(beliefs);
			decoder.backpropagate(observations);
			decoder.layer(0).backPropError(&layIn);
			beliefs.addScaled(decoder.learningRate(), layIn.error());
			beliefs.clip(-1.0, 1.0);
		}
	}
}


/// Decodes beliefs to predict observations
void ObservationModel::beliefsToObservations(const GVec& beliefs, GVec& observations)
{
	observations.resize(decoder.outputLayer().outputs());
	if(tutor)
		tutor->state_to_observations(beliefs, observations);
	else
	{
		decoder.forwardProp(beliefs);
		observations = decoder.outputLayer().activation();
	}
}


/// Encodes observations to predict beliefs
void ObservationModel::observationsToBeliefs(const GVec& observations, GVec& beliefs)
{
	beliefs.resize(encoder.outputLayer().outputs());
	if(tutor)
		tutor->observations_to_state(observations, beliefs);
	else
	{
		beliefs.put(0, observations, 0, beliefs.size());
		encoder.forwardProp(observations);
		beliefs = encoder.outputLayer().activation();
	}
}
