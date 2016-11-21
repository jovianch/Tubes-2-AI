
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 *
 * @author Joshua & Alif
 */
public class Main {
    private static int index = 0;
    /**
    * This method reads .arff file and convert it to dataset
    *
    * @param file : .arff file name and directory for making a dataset
    * @return dataset
    */
    public static Instances readDataset(String file) {
 
        // Initialize instances/dataset
        Instances dataset = null;
        try {
            // Read file
            BufferedReader breader = new BufferedReader(new FileReader(file));
            // Convert to Instances type
            dataset = new Instances(breader);

            String nama = "";
            int i;
            boolean found = false;
            for (i = 0;(i < dataset.numAttributes())&&(!found);i++) {
                nama = dataset.attribute(i).name();
                if (nama.equals("class")) {
                    found = true;
                    index = i;
                }
            }
            if (i == dataset.numAttributes()) {
                index = dataset.numAttributes()-1;
            }

            dataset.setClassIndex(index);

            // Print out the notification
            System.out.println("\nDataset have been read from iris.arff successfully\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
           return dataset;
    }
    
    /**
     * This method does data learning by full-training schema
     *
     * @param discretizedDataset : dataset that have been filtered by Discretize filter
         * @return fullTrainingClassifier : classifier after learning
     */
    public static Classifier learningFullTraining(Instances discretizedDataset) {

        // Initialize classifier
        Classifier fullTrainingClassifier = new NB();
        try {
            // Initialize Evaluation
            Evaluation fullTrainingEvaluation = new Evaluation(discretizedDataset);

            // Build classifier and evaluation
            fullTrainingClassifier.buildClassifier(discretizedDataset);
            fullTrainingEvaluation.evaluateModel(fullTrainingClassifier, discretizedDataset);

            // Print out the result of classifier model and evaluation
            System.out.println("\nData Learning Using Full-Training Schema\n");
            System.out.println(fullTrainingClassifier.toString());
            System.out.println(fullTrainingEvaluation.toSummaryString());
            System.out.println(fullTrainingEvaluation.toClassDetailsString());
            System.out.println(fullTrainingEvaluation.toMatrixString());

        } catch (Exception e) {
                e.printStackTrace();
        }
        return fullTrainingClassifier;
    }

    /**
     * This method does data learning by 10-fold cross validation schema
     *
     * @param discretizedDataset : dataset that have been filtered by Discretize filter
         * @return fullTrainingClassifier : classifier after learning
     */
    public static Classifier learningKFoldCrossValidation(Instances discretizedDataset) {
        // Initialize classifier
        Classifier kFoldCrossValidationClassifier = new NB();
        try {
            // Initialize evaluation
            Evaluation kFoldCrossValidationEvaluation = new Evaluation(discretizedDataset);

            // Build classifier and evaluation
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;

            kFoldCrossValidationEvaluation.crossValidateModel(kFoldCrossValidationClassifier, discretizedDataset, folds, rand);
            kFoldCrossValidationClassifier.buildClassifier(discretizedDataset);

            // Print out the result of classifier model and evaluation
            System.out.println("\nData Learning Using 10-Fold Cross Validation Schema\n");
            System.out.println(kFoldCrossValidationClassifier.toString());
            System.out.println(kFoldCrossValidationEvaluation.toSummaryString());
            System.out.println(kFoldCrossValidationEvaluation.toClassDetailsString());
            System.out.println(kFoldCrossValidationEvaluation.toMatrixString());

        } catch (Exception e) {
                e.printStackTrace();
        }
        return kFoldCrossValidationClassifier;
    }
        
    /**
     * This method creates instance from input user
     *
     * @param discretizedData : dataset that have been filtered by Discretize filter
     * @return instanceInput : instance that created by input user
     * @throws Exception 
     */
    public static Instance createInstanceFromInputUser(Instances discretizedData, Instances plainData){

        boolean needDisc = false;
        for (int i = 0;i < plainData.numAttributes();i++) {
            needDisc = needDisc | plainData.attribute(i).isNumeric();
        }
        System.out.println(needDisc);

        if (needDisc) {
            // Initialize instance
            Instance instanceInput = new DenseInstance(discretizedData.numAttributes()+1);

            // Equate attribute types to the dataset's
            instanceInput.setDataset(plainData);

            // Read value of attributes from user input
            for(int i=0; i<plainData.numAttributes(); i++){
                if(i!=plainData.classIndex()){
                    System.out.print("\nInput attribute " + instanceInput.attribute(i).name() + " : ");
                    Scanner s = new Scanner(System.in);
                    if(plainData.attribute(i).isNominal()) {
                        String attributeValueInput = s.next();
                        instanceInput.setValue(i, attributeValueInput);
                    } else {
                        Float attributeValueInput = s.nextFloat();
                        instanceInput.setValue(i, attributeValueInput);
                    }
                }
            }

            // Print out the notification
            System.out.println("\n\nInstance have been created successfully by user input\n");

            return instanceInput;
        }
        else {
            // Initialize instance
            Instance instanceInput = new DenseInstance(discretizedData.numAttributes()+1);

            // Equate attribute types to the dataset's
            instanceInput.setDataset(discretizedData);

            // Read value of attributes from user input
            for(int i=0; i<discretizedData.numAttributes(); i++){
                if(i!=discretizedData.classIndex()){
                    System.out.print("\nInput attribute " + instanceInput.attribute(i).name() + " : ");
                    Scanner s = new Scanner(System.in);
                    String attributeValueInput = s.next();
                    instanceInput.setValue(i, attributeValueInput);
                }
            }

            // Print out the notification
            System.out.println("\n\nInstance have been created successfully by user input\n");
            return instanceInput;
        }
    }
    
    public static Instance makeDiscretizeInstance(Discretize discretize, Instances dataset, Instance instanceInput) {
        Instance discretizeInstanceInput = null;
        try {
                // Set up option for filter -> default bin=4 and first until last attributes will be filtered
                String[] options = new String[2];
                options[0]="-R";
                options[1]="first-last";

                discretize.setOptions(options);
                discretize.setInputFormat(dataset);
                Filter.useFilter(dataset, discretize);

                if (discretize.input(instanceInput)) {
                        discretizeInstanceInput = discretize.output();
                }
        } catch (Exception e) {
                e.printStackTrace();
        }
        return discretizeInstanceInput;
    }
    
    public static void main(String[] argv) throws Exception {
        String filename = "";
        Scanner nama = new Scanner(System.in);
        filename = nama.next();
        
        Instances dataset = null;
        // Build Discretize filter
        Discretize discretize = new Discretize();

        dataset = readDataset(filename);
        NB wekatest12 = new NB();
        Instances preDisc = null;
        preDisc = readDataset(filename);
        
        wekatest12.buildClassifier(dataset);

        Classifier kfold = learningKFoldCrossValidation(wekatest12.dataset);
        Classifier fullt = learningFullTraining(wekatest12.dataset);
        
        System.out.println(wekatest12);
        
        Instance in = createInstanceFromInputUser(wekatest12.dataset, preDisc);
        //System.out.println("The instance is " + in);
        Instance fi = makeDiscretizeInstance(discretize, dataset, in);
        double x[] = wekatest12.distributionForInstance(fi);
        for(int k=0; k<wekatest12.n_index_value; k++){
            System.out.println(wekatest12.dataset.attribute(wekatest12.index).value(k) + " - " + x[k]);
        }
        int res = (int) wekatest12.classifyInstance(fi);
        System.out.println("Result: " + wekatest12.dataset.attribute(wekatest12.index).value(res));

        weka.core.SerializationHelper.write("NB.model", wekatest12);

        NB wekatest13 = new NB();
        wekatest13 = (NB) weka.core.SerializationHelper.read("NB.model");
        //print matriks nya
        System.out.println(wekatest13);
    }
}
