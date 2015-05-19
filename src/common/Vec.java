package common;

import common.json.JSONArray;
import java.util.Iterator;

/** Provides static methods for operating on arrays of doubles */
public class Vec
{
	public static void print(double[] vec) {
		System.out.print("[");
		if(vec.length > 0) {
			System.out.print(Double.toString(vec[0]));
			for(int i = 1; i < vec.length; i++) {
				System.out.print("," + Double.toString(vec[i]));
			}
		}
		System.out.print("]");
	}

	public static void println(double[] vec) {
		print(vec);
		System.out.println();
	}

	public static void setAll(double[] vec, double val) {
		for(int i = 0; i < vec.length; i++)
			vec[i] = val;
	}

	public static double squaredMagnitude(double[] vec) {
		double d = 0.0;
		for(int i = 0; i < vec.length; i++)
			d += vec[i] * vec[i];
		return d;
	}

	public static void normalize(double[] vec) {
		double mag = squaredMagnitude(vec);
		if(mag <= 0.0) {
			setAll(vec, 0.0);
			vec[0] = 1.0;
		} else {
			double s = 1.0 / Math.sqrt(mag);
			for(int i = 0; i < vec.length; i++)
				vec[i] *= s;
		}
	}

	public static void copy(double[] dest, double[] src) {
		if(dest.length != src.length)
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < src.length; i++) {
			dest[i] = src[i];
		}
	}

	public static double[] copy(double[] src) {
		double[] dest = new double[src.length];
		for(int i = 0; i < src.length; i++) {
			dest[i] = src[i];
		}
		return dest;
	}

	public static void add(double[] dest, double[] src) {
		if(dest.length != src.length)
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < dest.length; i++) {
			dest[i] += src[i];
		}
	}

	public static void scale(double[] dest, double scalar) {
		for(int i = 0; i < dest.length; i++) {
			dest[i] *= scalar;
		}
	}

	public static double dotProduct(double[] a, double[] b) {
		if(a.length != b.length)
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < a.length; i++)
			d += a[i] * b[i];
		return d;
	}

	public static double squaredDistance(double[] a, double[] b) {
		if(a.length != b.length)
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < a.length; i++) {
			double t = a[i] - b[i];
			d += t * t;
		}
		return d;
	}

	public static void clip(double[] vec, double min, double max) {
		if(max < min)
			throw new IllegalArgumentException("max must be >= min");
		for(int i = 0; i < vec.length; i++) {
			vec[i] = Math.max(min, Math.min(max, vec[i]));
		}
	}

	public static double[] concatenate(double[] a, double[] b) {
		double[] c = new double[a.length + b.length];
		for(int i = 0; i < a.length; i++)
			c[i] = a[i];
		for(int i = 0; i < b.length; i++)
			c[a.length + i] = b[i];
		return c;
	}

	public static double[] unmarshal(JSONArray arr) {
		Iterator<Double> it = arr.iterator();
		double[] v = new double[arr.size()];
		int i = 0;
		while(it.hasNext()) {
			v[i++] = it.next();
		}
		return v;
	}

	public static JSONArray marshal(double[] vec) {
		JSONArray v = new JSONArray();
		for(int i = 0; i < vec.length; i++)
			v.add(vec[i]);
		return v;
	}
}
