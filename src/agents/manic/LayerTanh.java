package agents.manic;

import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;
import common.Matrix;
import common.Vec;
import common.json.JSONObject;
import common.json.JSONArray;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.FileWriter;
import java.io.FileReader;


class LayerTanh {
	Matrix weights; // rows are inputs, cols are outputs
	double[] bias;
	double[] net;
	double[] activation;
	double[] error;
	//double[] hinge;


	LayerTanh(int inputs, int outputs) {
		weights = new Matrix();
		weights.setSize(inputs, outputs);
		bias = new double[outputs];
		net = new double[outputs];
		activation = new double[outputs];
		error = new double[outputs];
		//hinge = new double[outputs];
	}


	LayerTanh(LayerTanh that) {
		weights = new Matrix(that.weights);
		bias = Vec.copy(that.bias);
		net = Vec.copy(that.net);
		activation = Vec.copy(that.activation);
		error = Vec.copy(that.error);
		//hinge = Vec.copy(that.hinge);
	}


	LayerTanh(JSONObject obj) {
		weights = new Matrix((JSONObject)obj.get("weights"));
		bias = Vec.unmarshal((JSONArray)obj.get("bias"));
		net = new double[weights.cols()];
		activation = new double[weights.cols()];
		error = new double[weights.cols()];
		//hinge = unmarshalVector((JSONArray)obj.get("hinge"));
		if(bias.length != weights.cols() /*|| hinge.length != weights.cols()*/)
			throw new IllegalArgumentException("mismatching sizes");
	}


	/// Marshal this object into a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("weights", weights.marshal());
		obj.put("bias", Vec.marshal(bias));
		//obj.put("hinge", Vec.marshal(hinge));
		return obj;
	}


	void copy(LayerTanh src) {
		if(src.weights.rows() != weights.rows() || src.weights.cols() != weights.cols())
			throw new IllegalArgumentException("mismatching sizes");
		weights.setSize(0, src.weights.cols());
		weights.copyPart(src.weights, 0, 0, src.weights.rows(), src.weights.cols());
		for(int i = 0; i < bias.length; i++) {
			bias[i] = src.bias[i];
			//hinge[i] = src.hinge[i];
		}
	}


	int inputCount() { return weights.rows(); }
	int outputCount() { return weights.cols(); }


	void initWeights(Random r) {
		double dev = Math.max(0.3, 1.0 / weights.rows());
		for(int i = 0; i < weights.rows(); i++) {
			double[] row = weights.row(i);
			for(int j = 0; j < weights.cols(); j++) {
				row[j] = dev * r.nextGaussian();
			}
		}
		for(int j = 0; j < weights.cols(); j++) {
			bias[j] = dev * r.nextGaussian();
			//hinge[j] = 0.0;
		}
	}


	void feedForward(double[] in) {
		if(in.length != weights.rows())
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(in.length) + " != " + Integer.toString(weights.rows()));
		for(int i = 0; i < net.length; i++)
			net[i] = bias[i];
		for(int j = 0; j < weights.rows(); j++) {
			double v = in[j];
			double[] w = weights.row(j);
			for(int i = 0; i < weights.cols(); i++)
				net[i] += v * w[i];
		}
	}


	void feedForward2(double[] in1, double[] in2) {
		if(in1.length + in2.length != weights.rows())
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(in1.length) + " + " + Integer.toString(in2.length) + " != " + Integer.toString(weights.rows()));
		for(int i = 0; i < net.length; i++)
			net[i] = bias[i];
		for(int j = 0; j < in1.length; j++) {
			double v = in1[j];
			double[] w = weights.row(j);
			for(int i = 0; i < weights.cols(); i++)
				net[i] += v * w[i];
		}
		for(int j = 0; j < in2.length; j++) {
			double v = in2[j];
			double[] w = weights.row(in1.length + j);
			for(int i = 0; i < weights.cols(); i++)
				net[i] += v * w[i];
		}
	}


	void activate() {
		for(int i = 0; i < net.length; i++) {
			activation[i] = Math.tanh(net[i]);
			//activation[i] = hinge[i] * (Math.sqrt(net[i] * net[i] + 1) - 1) + net[i];
		}
	}


	void computeError(double[] target) {
		if(target.length != activation.length)
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(target.length) + " != " + Integer.toString(activation.length));
		for(int i = 0; i < activation.length; i++) {
			if(target[i] < -1.0 || target[i] > 1.0)
				throw new IllegalArgumentException("target value out of range for the tanh activation function");
			error[i] = target[i] - activation[i];
		}
	}


	void deactivate() {
		for(int i = 0; i < error.length; i++) {
			error[i] *= (1.0 - activation[i] * activation[i]);
			//error[i] *= (net[i] * hinge[i] / Math.sqrt(net[i] * net[i] + 1) + 1);
		}
	}


	void feedBack(double[] upstream) {
		if(upstream.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int j = 0; j < weights.rows(); j++) {
			double[] w = weights.row(j);
			double d = 0.0;
			for(int i = 0; i < weights.cols(); i++) {
				d += error[i] * w[i];
			}
			upstream[j] = d;
		}
	}


	void refineInputs(double[] inputs, double learningRate) {
		if(inputs.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int j = 0; j < weights.rows(); j++) {
			double[] w = weights.row(j);
			double d = 0.0;
			for(int i = 0; i < weights.cols(); i++) {
				d += error[i] * w[i];
			}
			inputs[j] += learningRate * d;
		}
	}


	void updateWeights(double[] in, double learningRate) {
		for(int i = 0; i < bias.length; i++) {
			bias[i] += learningRate * error[i];
		}
		for(int j = 0; j < weights.rows(); j++) {
			double[] w = weights.row(j);
			double x = learningRate * in[j];
			for(int i = 0; i < weights.cols(); i++) {
				w[i] += x * error[i];
			}
		}
	}

/*
	void bendHinge(double learningRate) {
		for(int i = 0; i < hinge.length; i++) {
			hinge[i] = Math.max(-1.0, Math.min(1.0, hinge[i] + learningRate * error[i] * (Math.sqrt(net[i] * net[i] + 1.0) - 1.0)));
		}
	}


	// Applies both L2 and L1 regularization to the hinge
	void straightenHinge(double lambda) {
		for(int i = 0; i < hinge.length; i++) {
			hinge[i] *= (1.0 - lambda);
			if(hinge[i] < 0.0)
				hinge[i] += lambda;
			else
				hinge[i] -= lambda;
		}
	}
*/

	// Applies both L2 and L1 regularization to the weights and bias values
	void regularizeWeights(double lambda) {
		for(int i = 0; i < weights.rows(); i++) {
			double[] row = weights.row(i);
			for(int j = 0; j < row.length; j++) {
				row[j] *= (1.0 - lambda);
				if(row[j] < 0.0)
					row[j] += lambda;
				else
					row[j] -= lambda;
			}
		}
		for(int j = 0; j < bias.length; j++) {
			bias[j] *= (1.0 - lambda);
			if(bias[j] < 0.0)
				bias[j] += lambda;
			else
				bias[j] -= lambda;
		}
	}
}
