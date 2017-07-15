// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Iterator;
import java.lang.StringBuilder;


/// This stores a matrix, A.K.A. data set, A.K.A. table. Each element is
/// represented as a double value. Nominal values are represented using their
/// corresponding zero-indexed enumeration value. For convenience,
/// the matrix also stores some meta-data which describes the columns (or attributes)
/// in the matrix.
public class Matrix
{
	/// Used to represent elements in the matrix for which the value is not known.
	public static final double UNKNOWN_VALUE = -1e308; 

	// Data
	private ArrayList<double[]> m_data = new ArrayList<double[]>(); //matrix elements

	// Meta-data
	private String m_filename;                          // the name of the file
	private ArrayList<String> m_attr_name;                 // the name of each attribute (or column)
	private ArrayList<HashMap<String, Integer>> m_str_to_enum; // value to enumeration
	private ArrayList<HashMap<Integer, String>> m_enum_to_str; // enumeration to value


	/// Creates a 0x0 matrix. (Next, to give this matrix some dimensions, you should call:
	///    loadARFF
	///    setSize
	///    addColumn, or
	///    copyMetaData
	@SuppressWarnings("unchecked")
	public Matrix() 
	{
		this.m_filename    = "";
		this.m_attr_name   = new ArrayList<String>();
		this.m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		this.m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
	}


	public Matrix(int rows, int cols)
	{
		this.m_filename    = "";
		this.m_attr_name   = new ArrayList<String>();
		this.m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		this.m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		setSize(rows, cols);
	}


	public Matrix(Matrix that)
	{
		m_filename = that.m_filename;
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		setSize(that.rows(), that.cols());
		copyBlock(0, 0, that, 0, 0, that.rows(), that.cols()); // (copies the meta data too)
	}


	public Matrix(Json n)
	{
		int rowCount = n.size();
		int colCount = n.get(0).size();
		setSize(rowCount, colCount);
		for(int i = 0; i < rowCount; i++)
		{
			double[] mrow = row(i);
			Json jrow = n.get(i);
			for(int j = 0; j < colCount; j++)
			{
				mrow[j] = jrow.getDouble(j);
			}
		}
	}


	public Json marshal()
	{
		Json list = Json.newList();
		for(int i = 0; i < rows(); i++)
			list.add(Vec.marshal(row(i)));
		return list;
	}


	/// Loads the matrix from an ARFF file
	public void loadARFF(String filename)
	{
		HashMap<String, Integer> tempMap  = new HashMap<String, Integer>(); //temp map for int->string map (attrInts)
		HashMap<Integer, String> tempMapS = new HashMap<Integer, String>(); //temp map for string->int map (attrString)
		int attrCount = 0; // Count number of attributes
		int lineNum = 0; // Used for exception messages
		Scanner s = null;
		m_str_to_enum.clear();
		m_enum_to_str.clear();
		m_attr_name.clear();

		try
		{
			s = new Scanner(new File(filename));
			while (s.hasNextLine())
			{
				lineNum++;
				String line  = s.nextLine().trim();
				String upper = line.toUpperCase();

				if (upper.startsWith("@RELATION"))
					m_filename = line.split(" ")[1];
				else if (upper.startsWith("@ATTRIBUTE"))
				{
					String[] pieces = line.split("\\s+");
					m_attr_name.add(pieces[1]);
					
					tempMap.clear();
					tempMapS.clear();
					
					// If the attribute is nominal
					if (pieces[2].startsWith("{"))
					{
						// Splits this string based on curly brackets or commas
						String[] attributeNames = pieces[2].split("[{},]");
						int valCount = 0;
						
						for (String attribute : attributeNames)
						{
							if (!attribute.equals("")) // Ignore empty strings
							{
								tempMapS.put(valCount, attribute);
								tempMap.put(attribute, valCount++);
							}
						}
					}
					
					// The attribute is continuous if it wasn't picked up in the previous "if" statement
					
					m_str_to_enum.add(new HashMap<String, Integer>(tempMap));
					m_enum_to_str.add(new HashMap<Integer, String>(tempMapS));
					
					attrCount++;
				}
				else if (upper.startsWith("@DATA"))
				{
					m_data.clear();
					
					while (s.hasNextLine())
					{
						double[] temp = new double[attrCount];

						lineNum++;
						line  = s.nextLine().trim();
						
						if (line.startsWith("%") || line.isEmpty()) continue;
						String[] pieces = line.split(",");
						
						if (pieces.length < attrCount) throw new IllegalArgumentException("Expected more elements on line: " + lineNum + ".");
						
						for (int i = 0; i < attrCount; i++)
						{
							int vals   = valueCount(i);
							String val = pieces[i];
							
							// Unknown values are always set to UNKNOWN_VALUE
							if (val.equals("?"))
							{
								temp[i] = UNKNOWN_VALUE;
								continue;
							}
		
							// If the attribute is nominal
							if (vals > 0)
							{
								HashMap<String, Integer> enumMap = m_str_to_enum.get(i);
								if (!enumMap.containsKey(val))
									throw new IllegalArgumentException("Unrecognized enumeration value " + val + " on line: " + lineNum + ".");
									
								temp[i] = (double)enumMap.get(val);
							}
							else
								temp[i] = Double.parseDouble(val); // The attribute is continuous
						}
						
						m_data.add(temp);
					}
				}
			}
		}
		catch (FileNotFoundException e)
		{
			throw new IllegalArgumentException("Failed to open file: " + filename + ".");
		}
		finally
		{
			s.close();
		}
	}


	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		for(int j = 0; j < rows(); j++)
		{
			if(j > 0)
				sb.append("\n");
			sb.append(Vec.toString(row(j)));
		}
		return sb.toString();
	}


	/// Saves the matrix to an ARFF file
	public void saveARFF(String filename)
	{		
		PrintWriter os = null;
		
		try
		{
			os = new PrintWriter(filename);
			// Print the relation name, if one has been provided ('x' is default)
			os.print("@RELATION ");
			os.println(m_filename.isEmpty() ? "x" : m_filename);
			
			// Print each attribute in order
			for (int i = 0; i < m_attr_name.size(); i++)
			{
				os.print("@ATTRIBUTE ");
				
				String attributeName = m_attr_name.get(i);
				os.print(attributeName.isEmpty() ? "x" : attributeName);
				
				int vals = valueCount(i);
				
				if (vals == 0) os.println(" REAL");
				else
				{
					os.print(" {");
					for (int j = 0; j < vals; j++)
					{
						os.print(attrValue(i, j));
						if (j + 1 < vals) os.print(",");
					}
					os.println("}");
				}
			}
			
			// Print the data
			os.println("@DATA");
			for (int i = 0; i < rows(); i++)
			{
				double[] row = m_data.get(i);
				for (int j = 0; j < cols(); j++)
				{
					if (row[j] == UNKNOWN_VALUE)
						os.print("?");
					else
					{
						int vals = valueCount(j);
						if (vals == 0) os.print(row[j]);
						else
						{
							int val = (int)row[j];
							if (val >= vals) throw new IllegalArgumentException("Value out of range.");
							os.print(attrValue(j, val));
						}
					}
					
					if (j + 1 < cols())	os.print(",");
				}
				os.println();
			}
		}
		catch (FileNotFoundException e)
		{
			throw new IllegalArgumentException("Error creating file: " + filename + ".");
		}
		finally
		{
			os.close();
		}
	}

	/// Makes a rows-by-columns matrix of *ALL CONTINUOUS VALUES*.
	/// This method wipes out any data currently in the matrix. It also
	/// wipes out any meta-data.
	public void setSize(int rows, int cols)
	{
		m_data.clear();

		// Set the meta-data
		m_filename = "";
		m_attr_name.clear();
		m_str_to_enum.clear();
		m_enum_to_str.clear();

		// Make space for each of the columns, then each of the rows
		newColumns(cols);
		newRows(rows);
	}

	/// Clears this matrix and copies the meta-data from that matrix.
	/// In other words, it makes a zero-row matrix with the same number
	/// of columns as "that" matrix. You will need to call newRow or newRows
	/// to give the matrix some rows.
	@SuppressWarnings("unchecked")
	public void copyMetaData(Matrix that)
	{
		m_data.clear();
		m_attr_name = new ArrayList<String>(that.m_attr_name);
		
		// Make a deep copy of that.m_str_to_enum
		m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		for (HashMap<String, Integer> map : that.m_str_to_enum)
		{
			HashMap<String, Integer> temp = new HashMap<String, Integer>();
			for (Map.Entry<String, Integer> entry : map.entrySet())
				temp.put(entry.getKey(), entry.getValue());
			
			m_str_to_enum.add(temp);
		}
		
		// Make a deep copy of that.m_enum_to_string
		m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		for (HashMap<Integer, String> map : that.m_enum_to_str)
		{
			HashMap<Integer, String> temp = new HashMap<Integer, String>();
			for (Map.Entry<Integer, String> entry : map.entrySet())
				temp.put(entry.getKey(), entry.getValue());
			
			m_enum_to_str.add(temp);
		}
	}

	/// Adds a column to this matrix with the specified number of values. (Use 0 for
	/// a continuous attribute.) This method also sets the number of rows to 0, so
	/// you will need to call newRow or newRows when you are done adding columns.
	public void newColumn(int vals)
	{
		m_data.clear();
		String name = "col_" + cols();
		
		m_attr_name.add(name);
		
		HashMap<String, Integer> temp_str_to_enum = new HashMap<String, Integer>();
		HashMap<Integer, String> temp_enum_to_str = new HashMap<Integer, String>();
		
		for (int i = 0; i < vals; i++)
		{
			String sVal = "val_" + i;
			temp_str_to_enum.put(sVal, i);
			temp_enum_to_str.put(i, sVal);
		}
		
		m_str_to_enum.add(temp_str_to_enum);
		m_enum_to_str.add(temp_enum_to_str);
	}
	

	/// Adds a column to this matrix with 0 values (continuous data).
	public void newColumn()
	{
		this.newColumn(0);
	}
	

	/// Adds n columns to this matrix, each with 0 values (continuous data).
	public void newColumns(int n)
	{
		for (int i = 0; i < n; i++)
			newColumn();
	}
	

	/// Adds one new row to this matrix. Returns a reference to the new row.
	public double[] newRow()
	{
		int c = cols();
		if (c == 0)
			throw new IllegalArgumentException("You must add some columns before you add any rows.");
		double[] newRow = new double[c];
		m_data.add(newRow);
		return newRow;
	}


	/// Adds one new row to this matrix at the specified location. Returns a reference to the new row.
	public double[] insertRow(int i)
	{
		int c = cols();
		if (c == 0)
			throw new IllegalArgumentException("You must add some columns before you add any rows.");
		double[] newRow = new double[c];
		m_data.add(i, newRow);
		return newRow;
	}


	/// Removes the specified row from this matrix. Returns a reference to the removed row.
	public double[] removeRow(int i)
	{
		return m_data.remove(i);
	}


	/// Appends the specified row to this matrix.
	public void takeRow(double[] row)
	{
		if(row.length != cols())
			throw new IllegalArgumentException("Row size differs from the number of columns in this matrix.");
		m_data.add(row);
	}


	/// Adds 'n' new rows to this matrix
	public void newRows(int n)
	{
		for (int i = 0; i < n; i++)
			newRow();
	}


	/// Returns the number of rows in the matrix
	public int rows() { return m_data.size(); }
	
	/// Returns the number of columns (or attributes) in the matrix
	public int cols() { return m_attr_name.size(); }
	
	/// Returns the name of the specified attribute
	public String attrName(int col) { return m_attr_name.get(col); }
	
	/// Returns the name of the specified value
	public String attrValue(int attr, int val)
	{		
		String value = m_enum_to_str.get(attr).get(val);
		if (value == null)
			throw new IllegalArgumentException("No name.");
		else return value;
	}
	
	/// Returns a reference to the specified row
	public double[] row(int index) { return m_data.get(index); }
	
	/// Swaps the positions of the two specified rows
	public void swapRows(int a, int b)
	{
		double[] temp = m_data.get(a);
		m_data.set(a, m_data.get(b));
		m_data.set(b, temp);
	}
	
	/// Returns the number of values associated with the specified attribute (or column)
	/// 0 = continuous, 2 = binary, 3 = trinary, etc.
	public int valueCount(int attr) { return m_enum_to_str.get(attr).size(); }
	
	/// Copies that matrix
	void copy(Matrix that)
	{
		setSize(that.rows(), that.cols());
		copyBlock(0, 0, that, 0, 0, that.rows(), that.cols());
	}

	/// Returns the mean of the elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double columnMean(int col)
	{
		double sum = 0.0;
		int count = 0;
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
			{
				sum += val;
				count++;
			}
		}
		
		return sum / count;
	}
	
	/// Returns the minimum element in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double columnMin(int col)
	{
		double min = Double.MAX_VALUE;
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
				min = Math.min(min, val);
		}
		
		return min;
	}

	/// Returns the maximum element in the specifed column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double columnMax(int col)
	{
		double max = -Double.MAX_VALUE;
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
				max = Math.max(max, val);
		}
		
		return max;
	}
	
	/// Returns the most common value in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double mostCommonValue(int col)
	{
		HashMap<Double, Integer> counts = new HashMap<Double, Integer>();
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
			{
				Integer result = counts.get(val);
				if (result == null) result = 0;
				
				counts.put(val, result + 1);
			}
		}
		
		int valueCount = 0;
		double value   = 0;
		for (Map.Entry<Double, Integer> entry : counts.entrySet())
		{
			if (entry.getValue() > valueCount)
			{
				value      = entry.getKey();
				valueCount = entry.getValue();
			}
		}
		
		return value;
	}

	/// Copies the specified rectangular portion of that matrix, and puts it in the specified location in this matrix.
	public void copyBlock(int destRow, int destCol, Matrix that, int rowBegin, int colBegin, int rowCount, int colCount)
	{
		if (destRow + rowCount > this.rows() || destCol + colCount > this.cols())
			throw new IllegalArgumentException("Out of range for destination matrix.");
		if (rowBegin + rowCount > that.rows() || colBegin + colCount > that.cols())
			throw new IllegalArgumentException("Out of range for source matrix.");

		// Copy the specified region of meta-data
		for (int i = 0; i < colCount; i++)
		{
			m_attr_name.set(destCol + i, that.m_attr_name.get(colBegin + i));
			m_str_to_enum.set(destCol + i, new HashMap<String, Integer>(that.m_str_to_enum.get(colBegin + i)));
			m_enum_to_str.set(destCol + i, new HashMap<Integer, String>(that.m_enum_to_str.get(colBegin + i)));
		}

		// Copy the specified region of data
		for (int i = 0; i < rowCount; i++)
		{
			double[] source = that.row(rowBegin + i);
			double[] dest = this.row(destRow + i);
			for(int j = 0; j < colCount; j++)
				dest[destCol + j] = source[colBegin + j];
		}
	}
	
	/// Sets every element in the matrix to the specified value.
	public void setAll(double val)
	{
		for (double[] vec : m_data)
		{
			for(int i = 0; i < vec.length; i++)
				vec[i] = val;
		}
	}


	/// Sets every element in the matrix to the specified value.
	public void scale(double scalar)
	{
		for (double[] vec : m_data)
		{
			for(int i = 0; i < vec.length; i++)
				vec[i] *= scalar;
		}
	}


	/// Adds every element in that matrix to this one
	public void addScaled(Matrix that, double scalar)
	{
		if(that.rows() != this.rows() || that.cols() != this.cols())
			throw new IllegalArgumentException("Mismatching size");
		for (int i = 0; i < rows(); i++)
		{
			double[] dest = this.row(i);
			double[] src = that.row(i);
			Vec.addScaled(dest, src, scalar);
		}
	}


	/// Sets this to the identity matrix.
	public void setToIdentity()
	{
		setAll(0.0);
		int m = Math.min(cols(), rows());
		for(int i = 0; i < m; i++)
			row(i)[i] = 1.0;
	}

	/// Throws an exception if that has a different number of columns than
	/// this, or if one of its columns has a different number of values.
	public void checkCompatibility(Matrix that)
	{
		int c = cols();
		if (that.cols() != c)
			throw new IllegalArgumentException("Matrices have different number of columns.");
		
		for (int i = 0; i < c; i++)
		{
			if (valueCount(i) != that.valueCount(i))
				throw new IllegalArgumentException("Column " + i + " has mis-matching number of values.");
		}
	}
}
