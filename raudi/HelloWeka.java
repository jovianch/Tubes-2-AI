
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;

import weka.*;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class HelloWeka {
	public static void main(String[] argv) throws Exception{
		try{
			//Read arff file dan simpen ke instances
			BufferedReader breader = new BufferedReader(new FileReader("iris.arff"));
			Instances dataset = new Instances(breader);
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			//Filtering -> discretize
			//Option -> default bin=10 dan attribute yang di filter first to last
			String[] options = new String[4];
			options[0]="-B"; options[1]="10";
			options[2]="-R"; options[3]="first-last";
			//Apply discretize
			Discretize discretize = new Discretize();
			discretize.setOptions(options);
			discretize.setInputFormat(dataset);
			Instances new_dataset = Filter.useFilter(dataset, discretize);
			new_dataset.setClassIndex(new_dataset.numAttributes()-1);
			
			//create classifier
			J48 tree = new J48();
			tree.buildClassifier(new_dataset);
			
			//save model in external file
			weka.core.SerializationHelper.write("gaib1.model", tree);
			
			//load model from external file
			J48 tree2 = (J48) weka.core.SerializationHelper.read("gaib1.model");
			
			//Buat instance baru sesuai input pengguna
			Instance input = new DenseInstance(5);
			//samain tipe attribute ny ma dataset
			input.setDataset(new_dataset);
			for(int i=0; i<dataset.classIndex(); i++){
				System.out.println("Input attribut ke-" + (i+1) + " :");
				Scanner s = new Scanner(System.in);
				float float_input = s.nextFloat();
				input.setValue(i, float_input);
			}

			//classify
			double result = tree2.classifyInstance(input);
			String result_string = input.classAttribute().value((int) result);

			System.out.println(new_dataset.toString());
			System.out.println(result_string);
			
			//dataset -> instance dari iris.arff
			//new_dataset -> dataset abis di filter (discretize)
			//input -> instance masukan user
			//result_string -> hasil classify dalam bentuk string
			//tree -> classifier yg dibikin pake dataset
			//tree2 -> classifier dari load model
		}
		catch(IOException e){
			e.printStackTrace();
		}
		//5.9,3,5.1,1.8
	}
}
