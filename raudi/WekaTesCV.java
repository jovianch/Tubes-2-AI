/**
 * Created by raudi on 10/30/16.
 */
import java.io.*;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.Filter;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaTesCV {

    public static void main(String[] args) throws Exception {
        // Create all Classifier
        Classifier allcls = new J48();
        Classifier allclsdis = new J48();
        Classifier cvcls = new J48();
        Classifier cvclsdis = new J48();

        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        DataSource source = new DataSource("iris.arff");

        // Create instances data from source
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Setting up Filter Discretize
        Discretize discrete = new Discretize();
        String[] options = new String[4];
        options[0] = "-R";
        options[1] = "first-last";
        options[2] = "-precision";
        options[3] = "6";
        discrete.setOptions(options);
        discrete.setInputFormat(data);

        // Create instances data after Discrete Filter
        Instances datadiscrete = Filter.useFilter(data, discrete);
        datadiscrete.setClassIndex(datadiscrete.numAttributes() - 1);

        // Create Evaluation from Classifier and Instances
        Evaluation alleval = new Evaluation(data);
        Evaluation allevaldis = new Evaluation(datadiscrete);
        Evaluation cveval = new Evaluation(data);
        Evaluation cvevaldis = new Evaluation(datadiscrete);

        // Print Model Classifier and Evaluation Result
        System.out.println("PAKE ALL DATA TRAINING\n");
        System.out.println("GAK PAKE DISCRETE\n");
        allcls.buildClassifier(data);
        alleval.evaluateModel(allcls, data);
        System.out.println(allcls.toString());
        System.out.println(alleval.toSummaryString());
        System.out.println(alleval.toClassDetailsString());
        System.out.println(alleval.toMatrixString());

        System.out.println("PAKE DISCRETE\n");
        allclsdis.buildClassifier(datadiscrete);
        allevaldis.evaluateModel(allclsdis, datadiscrete);
        System.out.println(allclsdis.toString());
        System.out.println(allevaldis.toSummaryString());
        System.out.println(allevaldis.toClassDetailsString());
        System.out.println(allevaldis.toMatrixString());

        System.out.println("PAKE CROSS VALIDATION 10 FOLDS\n");
        System.out.println("GAK PAKE DISCRETE\n");
        Random rand = new Random(1);  // using seed = 1
        int folds = 10;
        cveval.crossValidateModel(cvcls, data, folds, rand);
        cvcls.buildClassifier(data);
        System.out.println(cvcls.toString());
        System.out.println(cveval.toSummaryString());
        System.out.println(cveval.toClassDetailsString());
        System.out.println(cveval.toMatrixString());

        System.out.println("PAKE DISCRETE\n");
        cvevaldis.crossValidateModel(cvclsdis, datadiscrete, folds, rand);
        cvclsdis.buildClassifier(datadiscrete);
        System.out.println(cvclsdis.toString());
        System.out.println(cvevaldis.toSummaryString());
        System.out.println(cvevaldis.toClassDetailsString());
        System.out.println(cvevaldis.toMatrixString());

        //Serialization (Write model to external file)
        weka.core.SerializationHelper.write("j48.model", allclsdis);

        // Deserialization (Read model from external file)
        //Classifier cls = (J48) weka.core.SerializationHelper.read("j48.model");
        //System.out.println("Done");
        //System.out.println(cls.toString());

        // Create instance from user input
        Instance iExample = new DenseInstance(4);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String input;
        input = in.readLine();
        iExample.setValue(data.attribute(0), input);
        input = in.readLine();
        iExample.setValue(data.attribute(1), input);
        input = in.readLine();
        iExample.setValue(data.attribute(2), input);
        input = in.readLine();
        iExample.setValue(data.attribute(3), input);
        iExample.setDataset(data);

        // Classify instance from model
        double[] Result = allclsdis.distributionForInstance(iExample);
        // Probability class = "yes"
        System.out.println(Result[0]);
        // Probability class = "no"t
        System.out.println(Result[1]);

        /*  NULIS HASIL EVAL KE FILE, INI COBA2 DOANG GA PENTING,

            FileOutputStream out = new FileOutputStream("result.txt");
            ObjectOutputStream bw = new ObjectOutputStream(out);
            //bw.write("PAKE ALL DATA TRAINING\n\n");
            bw.writeObject(allclsdis.toString());
            bw.writeObject(allevaldis.toSummaryString() + "\n");
            bw.writeObject(allevaldis.toClassDetailsString() + "\n");
            bw.writeObject(allevaldis.toMatrixString() + "\n");
            byte[] allclsdisInBytes = allclsdis.toString().getBytes();
            byte[] allevaldisSumInBytes = allevaldis.toSummaryString().getBytes();
            byte[] allevaldisClassInBytes = allevaldis.toClassDetailsString().getBytes();
            byte[] allevaldisMatrixSumInBytes = allevaldis.toMatrixString().getBytes();
            bw.writeObject(allclsdisInBytes);
            bw.writeObject(allevaldisSumInBytes);
            bw.writeObject(allevaldisClassInBytes);
            bw.writeObject(allevaldisMatrixSumInBytes);
            bw.write("PAKE CROSS VALIDATION 10 FOLDS\n\n");
            bw.write(cvclsdis.toString());
            bw.write(cvevaldis.toSummaryString() + "\n");
            bw.write(cvevaldis.toClassDetailsString() + "\n");
            bw.write(cvevaldis.toMatrixString());

            bw.close();
        */
    }
}
