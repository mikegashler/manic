#include "TransitionModel.h"

/// General-purpose constructor
TransitionModel::TransitionModel(size_t input_dims, size_t output_dims, size_t total_layers, size_t queue_size, size_t trainItersPerPattern, GRand& r)
: rand(r),
trainInput(queue_size, input_dims),
trainOutput(queue_size, output_dims),
tutor(nullptr),
trainPos(0),
trainSize(0),
trainIters(trainItersPerPattern),
trainProgress(0),
err(0),
prevErr(0)
{
	size_t hidden = std::max((size_t)30, output_dims);
	model.addLayer(new GLayerClassic(input_dims, hidden));
	model.addLayer(new GLayerClassic(hidden, output_dims));
	GUniformRelation relIn(input_dims);
	GUniformRelation relOut(output_dims);
	model.beginIncrementalLearning(relIn, relOut);
	model.setLearningRate(0.03);
}


/// Unmarshaling constructor
TransitionModel::TransitionModel(GDomNode* pNode, GRand& r)
: rand(r),
model(pNode->field("model")),
trainInput(pNode->field("trainInput")),
trainOutput(pNode->field("trainOutput")),
tutor(nullptr),
trainPos(pNode->field("trainPos")->asInt()),
trainSize(pNode->field("trainSize")->asInt()),
trainIters(pNode->field("trainIters")->asInt()),
trainProgress(pNode->field("trainProgress")->asInt()),
err(pNode->field("err")->asDouble()),
prevErr(pNode->field("prevErr")->asDouble())
{
}


/// Marshals this model to a JSON DOM.
GDomNode* TransitionModel::marshal(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "model", model.serialize(pDoc));
	pNode->addField(pDoc, "trainPos", pDoc->newInt(trainPos));
	pNode->addField(pDoc, "trainSize", pDoc->newInt(trainSize));
	pNode->addField(pDoc, "trainIters", pDoc->newInt(trainIters));
	pNode->addField(pDoc, "trainInput", trainInput.serialize(pDoc));
	pNode->addField(pDoc, "trainOutput", trainOutput.serialize(pDoc));
	pNode->addField(pDoc, "trainProgress", pDoc->newInt(trainProgress));
	pNode->addField(pDoc, "err", pDoc->newDouble(err));
	pNode->addField(pDoc, "prevErr", pDoc->newDouble(prevErr));
	return pNode;
}


/// Returns the number of action dims
size_t TransitionModel::actionDims()
{
	return model.layer(0).inputs() - model.layer(model.layerCount() - 1).outputs();
}


/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
void TransitionModel::doSomeTraining()
{
	// Present one pattern
	double lambda = model.learningRate() * 0.0000001;
	model.scaleWeights(1.0 - lambda);
	model.diminishWeights(lambda);
	size_t index = rand.next(trainSize);
	model.trainIncremental(trainInput.row(index), trainOutput.row(index));
	err += trainOutput.row(index).squaredDistance(model.outputLayer().activation());

	// Measure how we are doing
	trainProgress++;
	if(trainProgress >= trainInput.rows()) {
		trainProgress = 0;
		prevErr = std::sqrt(err / trainInput.rows());
		err = 0.0;
		//std::cout << "Transition error:" << to_str(prevErr) << "\n";
	}
}


/// Refines this model based on a recently performed action and change in beliefs
void TransitionModel::trainIncremental(const GVec& beliefs, const GVec& actions, const GVec& nextBeliefs)
{
	// Buffer the pattern
	GVec& destIn = trainInput.row(trainPos);
	GVec& destOut = trainOutput.row(trainPos);
	trainPos++;
	trainSize = std::max(trainSize, trainPos);
	if(trainPos >= trainInput.rows())
		trainPos = 0;
	if(beliefs.size() + actions.size() != destIn.size() || beliefs.size() != destOut.size())
		throw Ex("size mismatch");
	destIn.put(0, beliefs);
	destIn.put(beliefs.size(), actions);
	for(size_t i = 0; i < destOut.size(); i++)
		destOut[i] = 0.5 * (nextBeliefs[i] - beliefs[i]);
/*
destIn.print();
std::cout << "->";
destOut.print();
std::cout << "\n";
std::cout << to_str(0.5 * cos(destIn[2])) << ", " << to_str(0.5 * sin(destIn[2])) << "\n";
*/
	// Refine the model
	size_t iters = std::min(trainIters, 1000 * trainSize);
	for(size_t i = 0; i < iters; i++)
		doSomeTraining();
}


/// Predict the belief vector that will result if the specified action is performed
void TransitionModel::anticipateNextBeliefs(const GVec& beliefs, const GVec& actions, GVec& anticipatedBeliefs)
{
	if(tutor)
		tutor->transition(beliefs, actions, anticipatedBeliefs);
	else
	{
		GAssert(beliefs.size() + actions.size() == model.layer(0).inputs());
		buf.resize(beliefs.size() + actions.size());
		buf.put(0, beliefs);
		buf.put(beliefs.size(), actions);
		model.forwardProp(buf);
		anticipatedBeliefs.copy(beliefs);
		anticipatedBeliefs.addScaled(2.0, model.outputLayer().activation());
		anticipatedBeliefs.clip(-1.0, 1.0);
	}
}


/// Compute the anticipated belief vector that will result if the specified plan is executed.
void TransitionModel::getFinalBeliefs(const GVec& beliefs, const GMatrix& plan, GVec& outFinalBeliefs)
{
	if(plan.rows() > 0)
		anticipateNextBeliefs(beliefs, plan[0], outFinalBeliefs);
	for(size_t i = 1; i < plan.rows(); i++) {
		anticipateNextBeliefs(outFinalBeliefs, plan[i], outFinalBeliefs);
	}
}
