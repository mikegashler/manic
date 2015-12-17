/// Represents a sequence of action vectors.
package agents.manic;

import java.util.ArrayList;
import java.util.Iterator;
import common.Vec;
import common.json.JSONArray;

public class Plan {
	public ArrayList<double[]> steps;


	// General-purpose constructor
	Plan() {
		steps = new ArrayList<double[]>();
	}

	// Copy constructor
	Plan(Plan that) {
		steps = new ArrayList<double[]>();
		for(int i = 0; i < that.size(); i++) {
			steps.add(Vec.copy(that.getActions(i)));
		}
	}

	/// Unmarshaling constructor
	Plan(JSONArray stepsArr) {
		steps = new ArrayList<double[]>();
		Iterator<JSONArray> it = stepsArr.iterator();
		while(it.hasNext()) {
			steps.add(Vec.unmarshal(it.next()));
		}
	}

	/// Marshals this model to a JSON DOM.
	JSONArray marshal() {
		JSONArray stepsArr = new JSONArray();
		for(int i = 0; i < steps.size(); i++) {
			stepsArr.add(Vec.marshal(steps.get(i)));
		}
		return stepsArr;
	}

	/// Returns the number of steps (or action vectors) in this plan
	int size() { return steps.size(); }

	/// Returns the ith action vector in this plan
	double[] getActions(int i) { return steps.get(i); }

	/// Prints a representation of the plan to stdout
	void print() {
		System.out.print("[");
		for(int i = 0; i < steps.size(); i++) {
			double[] actions = steps.get(i);
			System.out.print("(");
			for(int j = 0; j < actions.length; j++) {
				if(j > 0)
					System.out.print(",");
				System.out.print(Double.toString(actions[j]));
			}
			System.out.print(")");
		}
		System.out.println("]");
	}
}


