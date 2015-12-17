#include "ContentmentModel.h"
#include <iostream>
#include <GClasses/GActivation.h>

// General-purpose constructor
ContentmentModel::ContentmentModel(size_t beliefDims, size_t total_layers, size_t queue_size, size_t trainItersPerPattern, GRand& r)
: rand(r),
samples(queue_size, beliefDims),
contentment(queue_size, 1),
tutor(nullptr),
trainPos(0),
trainSize(0),
trainIters(0),
trainProgress(0),
err(0.0)
{
	// Init the model
	rand = r;
	int hidden = std::min((size_t)30, beliefDims * 10);
	model.addLayer(new GLayerClassic(beliefDims, hidden, new GActivationBend()));
	model.addLayer(new GLayerClassic(hidden, 1, new GActivationBend()));
	GUniformRelation relIn(beliefDims);
	GUniformRelation relOut(1);
	model.beginIncrementalLearning(relIn, relOut);

	// Init the meta-parameters
	trainIters = trainItersPerPattern;
	model.setLearningRate(0.03);
	targBuf.resize(1);
}


/// Unmarshaling constructor
ContentmentModel::ContentmentModel(GDomNode* obj, GRand& r)
: rand(r),
model(obj->field("model")),
samples(obj->field("samples")),
contentment(obj->field("contentment")),
tutor(nullptr),
trainPos(obj->field("trainPos")->asInt()),
trainSize(obj->field("trainSize")->asInt()),
trainIters(obj->field("trainIters")->asInt()),
trainProgress(obj->field("trainProgress")->asInt()),
err(obj->field("err")->asDouble())
{
	targBuf.resize(1);
}


/// Marshals this model to a JSON DOM.
GDomNode* ContentmentModel::marshal(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "model", model.serialize(pDoc));
	pNode->addField(pDoc, "samples", samples.serialize(pDoc));
	pNode->addField(pDoc, "contentment", contentment.serialize(pDoc));
	pNode->addField(pDoc, "trainPos", pDoc->newInt(trainPos));
	pNode->addField(pDoc, "trainSize", pDoc->newInt(trainSize));
	pNode->addField(pDoc, "trainIters", pDoc->newInt(trainIters));
	pNode->addField(pDoc, "trainProgress", pDoc->newInt(trainProgress));
	pNode->addField(pDoc, "err", pDoc->newDouble(err));
	return pNode;
}


/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
void ContentmentModel::doSomeTraining()
{
	// Present a sample of beliefs and corresponding contentment for training
	size_t index = rand.next(trainSize);
	double lambda = model.learningRate() * 0.000001;
	model.scaleWeights(1.0 - lambda);
	model.diminishWeights(lambda);
	model.trainIncremental(samples.row(index), contentment.row(index));
	err += contentment.row(index).squaredDistance(model.outputLayer().activation());
	if(++trainProgress >= 1000)
	{
		trainProgress = 0;
		//std::cout << "Contentment error: " << to_str(err / 1000.0) << "\n";
		err = 0.0;
	}
}


/// Refines this model based on feedback from the mentor
void ContentmentModel::trainIncremental(const GVec& sample_beliefs, double sample_contentment)
{
	// Buffer the samples
	GVec& dest = samples.row(trainPos);
	if(sample_beliefs.size() != dest.size())
		throw Ex("size mismatch");
	dest = sample_beliefs;
	contentment.row(trainPos)[0] = sample_contentment;
	trainPos++;
	trainSize = std::max(trainSize, trainPos);
	if(trainPos >= samples.rows())
		trainPos = 0;

	// Do a few iterations of stochastic gradient descent
	size_t iters = std::min(trainIters, trainSize);
	for(size_t i = 0; i < iters; i++)
		doSomeTraining();
}


/// Computes the contentment of a particular belief vector
double ContentmentModel::evaluate(const GVec& beliefs)
{
	if(tutor)
		return tutor->evaluate_state(beliefs);
	else
	{
		model.forwardProp(beliefs);
		return model.outputLayer().activation()[0];
	}
}
