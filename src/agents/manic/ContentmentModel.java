package agents.manic;

import common.Matrix;
import common.json.JSONObject;
import java.util.Random;

/// A model that maps from anticipated beliefs to contentment (or utility).
/// This model is trained by reinforcement from a teacher.
public class ContentmentModel {
	public Random rand;
	public NeuralNet model;
	public Matrix samples;
	public Matrix contentment;
	public int trainPos;
	public int trainSize;
	public int trainIters;
	public double learningRate;
	double[] targBuf;


	// General-purpose constructor
	ContentmentModel(int beliefDims, int total_layers, int queue_size, int trainItersPerPattern, Random r) {

		// Init the model
		rand = r;
		model = new NeuralNet();
		int hidden = Math.min(30, beliefDims * 10);
		model.layers.add(new LayerTanh(beliefDims, hidden));
		model.layers.add(new LayerTanh(hidden, 1));
		model.init(rand);

		// Init the buffers
		samples = new Matrix(queue_size, beliefDims);
		contentment = new Matrix(queue_size, 1);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.03;
		targBuf = new double[1];
	}


	/// Unmarshaling constructor
	ContentmentModel(JSONObject obj, Random r) {
		rand = r;
		model = new NeuralNet((JSONObject)obj.get("model"));
		samples = new Matrix((JSONObject)obj.get("samples"));
		contentment = new Matrix((JSONObject)obj.get("contentment"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		learningRate = (Double)obj.get("learningRate");
		targBuf = new double[1];
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("model", model.marshal());
		obj.put("samples", samples.marshal());
		obj.put("contentment", contentment.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("trainIters", trainIters);
		obj.put("learningRate", learningRate);
		return obj;
	}


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() {

		// Present a sample of beliefs and corresponding contentment for training
		int index = rand.nextInt(trainSize);
		model.regularize(learningRate, 0.000001);
		model.trainIncremental(samples.row(index), contentment.row(index), learningRate);
	}


	/// Refines this model based on feedback from the teacher
	void trainIncremental(double[] sample_beliefs, double sample_contentment) {

		// Buffer the samples
		double[] dest = samples.row(trainPos);
		if(sample_beliefs.length != dest.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < dest.length; i++)
			dest[i] = sample_beliefs[i];
		contentment.row(trainPos)[0] = sample_contentment;
		trainPos++;
		trainSize = Math.max(trainSize, trainPos);
		if(trainPos >= samples.rows())
			trainPos = 0;

		// Do a few iterations of stochastic gradient descent
		int iters = Math.min(trainIters, trainSize);
		for(int i = 0; i < iters; i++)
			doSomeTraining();
	}


	/// Computes the contentment of a particular belief vector
	public double evaluate(double[] beliefs) {
		double[] output = model.forwardProp(beliefs);
		return output[0];
	}
}


