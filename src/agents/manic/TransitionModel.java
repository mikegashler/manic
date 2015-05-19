package agents.manic;

import java.util.Random;
import common.Matrix;
import common.Vec;
import common.json.JSONObject;

/// A model that maps from current beliefs and actions to anticipated beliefs.
/// This model is trained in a supervised manner.
public class TransitionModel {
	Random rand;
	NeuralNet model;
	Matrix trainInput;
	Matrix trainOutput;
	int trainPos;
	public int trainSize;
	int trainIters;
	int trainProgress;
	double learningRate;
	double err;
	double prevErr;


	/// General-purpose constructor
	TransitionModel(int input_dims, int output_dims, int total_layers, int queue_size, int trainItersPerPattern, Random r) {

		// Init the model
		rand = r;
		model = new NeuralNet();
		int hidden = Math.max(30, output_dims);
		model.layers.add(new LayerTanh(input_dims, hidden));
		model.layers.add(new LayerTanh(hidden, output_dims));
		model.init(rand);

		// Init the buffers
		trainInput = new Matrix(queue_size, input_dims);
		trainOutput = new Matrix(queue_size, output_dims);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.03;
	}


	/// Unmarshaling constructor
	TransitionModel(JSONObject obj, Random r) {
		rand = r;
		model = new NeuralNet((JSONObject)obj.get("model"));
		trainInput = new Matrix((JSONObject)obj.get("trainInput"));
		trainOutput = new Matrix((JSONObject)obj.get("trainOutput"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		trainProgress = ((Long)obj.get("trainProgress")).intValue();
		learningRate = (Double)obj.get("learningRate");
		err = (Double)obj.get("err");
		prevErr = (Double)obj.get("prevErr");
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("model", model.marshal());
		obj.put("trainInput", trainInput.marshal());
		obj.put("trainOutput", trainOutput.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("trainIters", trainIters);
		obj.put("trainProgress", trainProgress);
		obj.put("learningRate", learningRate);
		obj.put("err", err);
		obj.put("prevErr", prevErr);
		return obj;
	}


	/// Returns the number of action dims
	int actionDims() { return model.layers.get(0).inputCount() - model.layers.get(model.layers.size() - 1).outputCount(); }


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() {

		// Present one pattern
		model.regularize(learningRate, 0.0000001);
		int index = rand.nextInt(trainSize);
		model.trainIncremental(trainInput.row(index), trainOutput.row(index), learningRate);
		err += Vec.squaredDistance(model.layers.get(model.layers.size() - 1).activation, trainOutput.row(index));

		// Measure how we are doing
		trainProgress++;
		if(trainProgress >= trainInput.rows()) {
			trainProgress = 0;
			prevErr = Math.sqrt(err / trainInput.rows());
		}
	}


	/// Refines this model based on a recently performed action and change in beliefs
	void trainIncremental(double[] beliefs, double[] actions, double[] nextBeliefs) {

		// Buffer the pattern
		double[] destIn = trainInput.row(trainPos);
		double[] destOut = trainOutput.row(trainPos);
		trainPos++;
		trainSize = Math.max(trainSize, trainPos);
		if(trainPos >= trainInput.rows())
			trainPos = 0;
		if(beliefs.length + actions.length != destIn.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < beliefs.length; i++)
			destIn[i] = beliefs[i];
		for(int i = 0; i < actions.length; i++)
			destIn[beliefs.length + i] = actions[i];
		for(int i = 0; i < destOut.length; i++)
			destOut[i] = nextBeliefs[i] - beliefs[i];

		// Refine the model
		int iters = Math.min(trainIters, 100 * trainSize);
		for(int i = 0; i < iters; i++)
			doSomeTraining();
	}


	/// Predict the belief vector that will result if the specified action is performed
	public void anticipateNextBeliefsInPlace(double[] beliefs, double[] actions, double[] anticipatedBeliefs) {
		double[] pred = model.forwardProp2(beliefs, actions);
		for(int i = 0; i < pred.length; i++) {
			anticipatedBeliefs[i] = Math.max(-1.0, Math.min(1.0, beliefs[i] + pred[i]));
		}
	}


	/// Predict the belief vector that will result if the specified action is performed
	public double[] anticipateNextBeliefs(double[] beliefs, double[] actions) {
		double[] anticipatedBeliefs = new double[beliefs.length];
		anticipateNextBeliefsInPlace(beliefs, actions, anticipatedBeliefs);
		return anticipatedBeliefs;
	}


	/// Compute the anticipated belief vector that will result if the specified plan is executed.
	public double[] getFinalBeliefs(double[] beliefs, Plan plan) {
		for(int i = 0; i < plan.size(); i++) {
			beliefs = anticipateNextBeliefs(beliefs, plan.getActions(i));
		}
		return beliefs;
	}
}
