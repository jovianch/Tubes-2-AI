
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
import weka.filters.unsupervised.attribute.Remove;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 *
 * @author Joshua & Alif
 */
public class FFNNMain {
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
            int i, idxdel;
            boolean found = false;
            for (i = 0;(i < dataset.numAttributes())&&(!found);i++) {
                nama = dataset.attribute(i).name();
                if (nama.equals("Walc")) {
                    found = true;
                    index = i;
                }
            }
            idxdel = index;
            if (i == dataset.numAttributes()) {
                index = dataset.numAttributes()-1;
            }

            dataset.setClassIndex(index);

            String idxdel_str = Integer.toString(idxdel);
            System.out.println(idxdel_str);
            Remove remove = new Remove();
            remove.setAttributeIndices(idxdel_str);
            try {
                remove.setInputFormat(dataset);
            } catch (Exception e) {
                e.printStackTrace();
            }

            Instances instNew = Filter.useFilter(dataset, remove);
            //instNew.setClassIndex(dataset.classIndex());
            // Print out the notification
            System.out.println("\nDataset have been read from iris.arff successfully\n");
            dataset = instNew;
            System.out.println(dataset.classIndex());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return dataset;
    }
    
    /**
     * This method does data learning by full-training schema
     *
     * @param  : dataset that have been filtered by Discretize filter
         * @return fullTrainingClassifier : classifier after learning
     */
    public static void learningFullTraining(Classifier fullTrainingClassifier, Instances dataset) {
        // Initialize classifier

        try {
            // Initialize Evaluation
            Evaluation fullTrainingEvaluation = new Evaluation(dataset);


            fullTrainingEvaluation.evaluateModel(fullTrainingClassifier, dataset);

            // Print out the result of classifier model and evaluation
            System.out.println("\nData Learning Using Full-Training Schema\n");
            System.out.println(fullTrainingClassifier.toString());
            System.out.println(fullTrainingEvaluation.toSummaryString());
            System.out.println(fullTrainingEvaluation.toClassDetailsString());
            System.out.println(fullTrainingEvaluation.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * This method does data learning by 10-fold cross validation schema
     *
     * @param  : dataset that have been filtered by Discretize filter
         * @return fullTrainingClassifier : classifier after learning
     */
    public static void learningKFoldCrossValidation(Classifier kFoldClassifier, Instances dataset) {

        try {
            // Initialize evaluation
            Evaluation kFoldCrossValidationEvaluation = new Evaluation(dataset);

            // Build classifier and evaluation
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;

            kFoldCrossValidationEvaluation.crossValidateModel(kFoldClassifier, dataset, folds, rand);

            // Print out the result of classifier model and evaluation
            System.out.println("\nData Learning Using 10-Fold Cross Validation Schema\n");
            System.out.println(kFoldClassifier.toString());
            System.out.println(kFoldCrossValidationEvaluation.toSummaryString());
            System.out.println(kFoldCrossValidationEvaluation.toClassDetailsString());
            System.out.println(kFoldCrossValidationEvaluation.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }

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
        // Baca file arff
        boolean save = true, read = true;
        String filename = "";
        System.out.print("Enter filename: ");
        Scanner nama = new Scanner(System.in);
        filename = nama.next();
                
        Instances dataset = null;
        dataset = readDataset(filename);
        Instances preDisc = null;
        preDisc = readDataset(filename);

        FFNN neural2 = new FFNN();
        if (read) {
            neural2 = (FFNN) weka.core.SerializationHelper.read("FFNN3v1.model");
            System.out.println(neural2);
        }

        //FFNN fullt = new FFNN();
        // Build classifier and evaluation
        /*String[] options = new String[6];
        options[0]="-H";
        options[1]="37";
        options[2]="-L";
        options[3]="1";
        options[4]="-N";
        options[5]="300";
        neural2.setOptions(options);
        neural2.buildClassifier(dataset);*/
        double updateValue;
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < dataset.numAttributes(); j++) {
                //Normalization
                if (dataset.classIndex() != j) {
                    updateValue = dataset.instance(i).value(j) / neural2.getRange(j);
                    dataset.instance(i).setValue(j, updateValue);
                }
            }
        }
        //learningKFoldCrossValidation(neural2, dataset);
        learningFullTraining(neural2, dataset);

        /*if (save) {
            weka.core.SerializationHelper.write("FFNN3v2.model", neural2);
            System.out.println("Save successfully");
            //weka.core.SerializationHelper.write("FFNNKF.model", FFNNKF);
        }*/
        Instance in = createInstanceFromInputUser(neural2.getInstances(),dataset);

        for (int j = 0; j < neural2.getInstances().numAttributes(); j++) {
            //Normalization
            if (neural2.getInstances().classIndex() != j) {
                updateValue = in.value(j) / neural2.getRange(j);
                in.setValue(j, updateValue);
            }
        }
        //System.out.println(neural1);

        double x[] = neural2.distributionForInstance(in);
        for(int k=0; k < neural2.getInstances().numClasses(); k++){
            System.out.println(x[k]);
        }

        int res = (int) neural2.classifyInstance(in);

        System.out.println("Result: " + neural2.getInstances().attribute(neural2.getInstances().classIndex()).value(res));

    }
}
