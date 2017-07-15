import java.util.Random;

abstract class Layer
{
	double[] activation;
	double[] error;

	static final int t_linear = 0;
	static final int t_tanh = 1;


	/// General-purpose constructor
	Layer(int outputs)
	{
		activation = new double[outputs];
		error = new double[outputs];
	}


	/// Copy constructor
	Layer(Layer that)
	{
		activation = Vec.copy(that.activation);
		error = Vec.copy(that.error);
	}


	/// Unmarshal from a JSON DOM
	Layer(Json n)
	{
		int units = (int)n.getLong("units");
		activation = new double[units];
		error = new double[units];
	}


	void computeError(double[] target)
	{
		if(target.length != activation.length)
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(target.length) + " != " + Integer.toString(activation.length));
		for(int i = 0; i < activation.length; i++)
		{
			error[i] = target[i] - activation[i];
		}
	}


	int outputCount()
	{
		return activation.length;
	}


	static Layer unmarshal(Json n)
	{
		int t = (int)n.getLong("type");
		switch(t)
		{
			case t_linear: return new LayerLinear(n);
			case t_tanh: return new LayerTanh(n);
			default: throw new RuntimeException("Unrecognized type");
		}
	}


	protected abstract Layer clone();
	abstract Json marshal();
	abstract int type();
	abstract int inputCount();
	abstract void initWeights(Random r);
	abstract double[] forwardProp(double[] in);
	abstract void backProp(Layer upStream);
	abstract void scaleGradient(double momentum);
	abstract void updateGradient(double[] in);
	abstract void step(double stepSize);
	abstract int countWeights();
	abstract int setWeights(double[] w, int start);
	abstract void regularizeWeights(double lambda);
}



class LayerLinear extends Layer
{
	Matrix weights; // rows are inputs, cols are outputs
	Matrix weightsGrad;
	double[] bias;
	double[] biasGrad;


	/// General-purpose constructor
	LayerLinear(int inputs, int outputs)
	{
		super(outputs);
		weights = new Matrix();
		weights.setSize(inputs, outputs);
		weightsGrad = new Matrix();
		weightsGrad.setSize(inputs, outputs);
		bias = new double[outputs];
		biasGrad = new double[outputs];
	}


	/// Copy constructor
	LayerLinear(LayerLinear that)
	{
		super(that);
		weights = new Matrix(that.weights);
		weightsGrad = new Matrix(that.weightsGrad);
		bias = Vec.copy(that.bias);
		biasGrad = Vec.copy(that.biasGrad);
		weightsGrad = new Matrix();
		weightsGrad.setSize(weights.rows(), weights.cols());
		weightsGrad.setAll(0.0);
		biasGrad = new double[weights.cols()];
		Vec.setAll(biasGrad, 0.0);
	}


	/// Unmarshal from a JSON DOM
	LayerLinear(Json n)
	{
		super(n);
		weights = new Matrix(n.get("weights"));
		bias = Vec.unmarshal(n.get("bias"));
	}


	protected LayerLinear clone()
	{
		return new LayerLinear(this);
	}


	/// Marshal into a JSON DOM
	Json marshal()
	{
		Json ob = Json.newObject();
		ob.add("units", (long)outputCount()); // required in all layers
		ob.add("weights", weights.marshal());
		ob.add("bias", Vec.marshal(bias));
		return ob;
	}


	void copy(LayerLinear src)
	{
		if(src.weights.rows() != weights.rows() || src.weights.cols() != weights.cols())
			throw new IllegalArgumentException("mismatching sizes");
		weights.copyBlock(0, 0, src.weights, 0, 0, src.weights.rows(), src.weights.cols());
		for(int i = 0; i < bias.length; i++)
		{
			bias[i] = src.bias[i];
		}
	}


	int type() { return t_linear; }
	int inputCount() { return weights.rows(); }


	void initWeights(Random r)
	{
		double dev = Math.max(0.3, 1.0 / weights.rows());
		for(int i = 0; i < weights.rows(); i++)
		{
			double[] row = weights.row(i);
			for(int j = 0; j < weights.cols(); j++)
			{
				row[j] = dev * r.nextGaussian();
			}
		}
		for(int j = 0; j < weights.cols(); j++) {
			bias[j] = dev * r.nextGaussian();
		}
		weightsGrad.setAll(0.0);
		Vec.setAll(biasGrad, 0.0);
	}


	int countWeights()
	{
		return weights.rows() * weights.cols() + bias.length;
	}


	int setWeights(double[] w, int start)
	{
		int oldStart = start;
		for(int i = 0; i < bias.length; i++)
			bias[i] = w[start++];
		for(int i = 0; i < weights.rows(); i++)
		{
			double[] row = weights.row(i);
			for(int j = 0; j < weights.cols(); j++)
				row[j] = w[start++];
		}
		return start - oldStart;
	}


	double[] forwardProp(double[] in)
	{
		if(in.length != weights.rows())
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(in.length) + " != " + Integer.toString(weights.rows()));
		for(int i = 0; i < activation.length; i++)
			activation[i] = bias[i];
		for(int j = 0; j < weights.rows(); j++)
		{
			double v = in[j];
			double[] w = weights.row(j);
			for(int i = 0; i < weights.cols(); i++)
				activation[i] += v * w[i];
		}
		return activation;
	}


	double[] forwardProp2(double[] in1, double[] in2)
	{
		if(in1.length + in2.length != weights.rows())
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(in1.length) + " + " + Integer.toString(in2.length) + " != " + Integer.toString(weights.rows()));
		for(int i = 0; i < activation.length; i++)
			activation[i] = bias[i];
		for(int j = 0; j < in1.length; j++)
		{
			double v = in1[j];
			double[] w = weights.row(j);
			for(int i = 0; i < weights.cols(); i++)
				activation[i] += v * w[i];
		}
		for(int j = 0; j < in2.length; j++)
		{
			double v = in2[j];
			double[] w = weights.row(in1.length + j);
			for(int i = 0; i < weights.cols(); i++)
				activation[i] += v * w[i];
		}
		return activation;
	}


	void backProp(Layer upStream)
	{
		if(upStream.outputCount() != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int j = 0; j < weights.rows(); j++)
		{
			double[] w = weights.row(j);
			double d = 0.0;
			for(int i = 0; i < weights.cols(); i++)
			{
				d += error[i] * w[i];
			}
			upStream.error[j] = d;
		}
	}


	void refineInputs(double[] inputs, double learningRate)
	{
		if(inputs.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int j = 0; j < weights.rows(); j++)
		{
			double[] w = weights.row(j);
			double d = 0.0;
			for(int i = 0; i < weights.cols(); i++)
			{
				d += error[i] * w[i];
			}
			inputs[j] += learningRate * d;
		}
	}


	void scaleGradient(double momentum)
	{
		weightsGrad.scale(momentum);
		Vec.scale(biasGrad, momentum);
	}


	void updateGradient(double[] in)
	{
		for(int i = 0; i < bias.length; i++)
		{
			biasGrad[i] += error[i];
		}
		for(int j = 0; j < weights.rows(); j++)
		{
			double[] w = weightsGrad.row(j);
			double x = in[j];
			for(int i = 0; i < weights.cols(); i++)
			{
				w[i] += x * error[i];
			}
		}
	}


	void step(double stepSize)
	{
		weights.addScaled(weightsGrad, stepSize);
		Vec.addScaled(bias, biasGrad, stepSize);
	}


	// Applies both L2 and L1 regularization to the weights and bias values
	void regularizeWeights(double lambda)
	{
		for(int i = 0; i < weights.rows(); i++)
		{
			double[] row = weights.row(i);
			for(int j = 0; j < row.length; j++)
			{
				row[j] *= (1.0 - lambda);
				if(row[j] < 0.0)
					row[j] += lambda;
				else
					row[j] -= lambda;
			}
		}
		for(int j = 0; j < bias.length; j++)
		{
			bias[j] *= (1.0 - lambda);
			if(bias[j] < 0.0)
				bias[j] += lambda;
			else
				bias[j] -= lambda;
		}
	}
}





class LayerTanh extends Layer
{
	/// General-purpose constructor
	LayerTanh(int nodes)
	{
		super(nodes);
	}


	/// Copy constructor
	LayerTanh(LayerTanh that)
	{
		super(that);
	}


	/// Unmarshal from a JSON DOM
	LayerTanh(Json n)
	{
		super(n);
	}


	protected LayerTanh clone()
	{
		return new LayerTanh(this);
	}


	/// Marshal into a JSON DOM
	Json marshal()
	{
		Json ob = Json.newObject();
		ob.add("units", (long)outputCount()); // required in all layers
		return ob;
	}


	void copy(LayerTanh src)
	{
	}


	int type() { return t_tanh; }
	int inputCount() { return activation.length; }


	void initWeights(Random r)
	{
	}


	int countWeights()
	{
		return 0;
	}


	int setWeights(double[] w, int start)
	{
		if(w.length != 0)
			throw new IllegalArgumentException("size mismatch");
		return 0;
	}


	double[] forwardProp(double[] in)
	{
		if(in.length != outputCount())
			throw new IllegalArgumentException("size mismatch. " + Integer.toString(in.length) + " != " + Integer.toString(outputCount()));
		for(int i = 0; i < activation.length; i++)
		{
			activation[i] = Math.tanh(in[i]);
		}
		return activation;
	}


	void backProp(Layer upStream)
	{
		if(upStream.outputCount() != outputCount())
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < activation.length; i++)
		{
			upStream.error[i] = error[i] * (1.0 - activation[i] * activation[i]);
		}
	}


	void scaleGradient(double momentum)
	{
	}


	void updateGradient(double[] in)
	{
	}


	void step(double stepSize)
	{
	}


	// Applies both L2 and L1 regularization to the weights and bias values
	void regularizeWeights(double lambda)
	{
	}
}

