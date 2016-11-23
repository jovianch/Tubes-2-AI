
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

        double updateValue;

        Instances datasetFT = null;
        datasetFT = readDataset(filename);
        Instances datasetKF = null;
        datasetKF = readDataset(filename);
        Instances datasetST = null;
        datasetST = readDataset(filename);

        FFNN classifierFT = new FFNN();
        FFNN classifierKF = new FFNN();
        FFNN classifierST = new FFNN();
        if (read) {
            classifierFT = (FFNN) weka.core.SerializationHelper.read("FFNN-Student-Wacl-FT.model");
            System.out.println(classifierFT);
            classifierKF = (FFNN) weka.core.SerializationHelper.read("FFNN-Student-Wacl-KF.model");
            System.out.println(classifierKF);
            classifierST = (FFNN) weka.core.SerializationHelper.read("FFNN-Student-Wacl-ST.model");
            System.out.println(classifierST);
        }

        // Build classifier and evaluation
        /*String[] options = new String[6];
        options[0]="-H";
        options[1]="37";
        options[2]="-L";
        options[3]="1";
        options[4]="-N";
        options[5]="300";
        classifierFT.setOptions(options);
        classifierFT.buildClassifier(datasetFT);*/

        //NORMALIZE
        for (int i = 0; i < datasetFT.size(); i++) {
            for (int j = 0; j < datasetFT.numAttributes(); j++) {
                //Normalization
                if (datasetFT.classIndex() != j) {
                    updateValue = datasetFT.instance(i).value(j) / classifierFT.getRange(j);
                    datasetFT.instance(i).setValue(j, updateValue);
                }
            }
        }
        for (int i = 0; i < datasetKF.size(); i++) {
            for (int j = 0; j < datasetKF.numAttributes(); j++) {
                //Normalization
                if (datasetKF.classIndex() != j) {
                    updateValue = datasetKF.instance(i).value(j) / classifierKF.getRange(j);
                    datasetKF.instance(i).setValue(j, updateValue);
                }
            }
        }

        //SPLIT TEST
        datasetST.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(datasetST.numInstances() * 0.8);
        int testSize = datasetST.numInstances() - trainSize;
        Instances train = new Instances(datasetST, 0, trainSize);
        Instances test = new Instances(datasetST, trainSize, testSize);

        String[] optionsST = new String[6];
        optionsST[0]="-H";
        optionsST[1]="37";
        optionsST[2]="-L";
        optionsST[3]="1";
        optionsST[4]="-N";
        optionsST[5]="300";
        classifierST.setOptions(optionsST);
        classifierST.buildClassifier(train);
        for (int i = 0; i < test.size(); i++) {
            for (int j = 0; j < test.numAttributes(); j++) {
                //Normalization
                if (test.classIndex() != j) {
                    updateValue = test.instance(i).value(j) / classifierST.getRange(j);
                    test.instance(i).setValue(j, updateValue);
                }
            }
        }
        //SPLIT TEST

        learningFullTraining(classifierFT, datasetFT);
        learningKFoldCrossValidation(classifierKF, datasetKF);
        learningFullTraining(classifierST, test);


        /*if (save) {
            weka.core.SerializationHelper.write("FFNN8v1.model", classifierFT);
            System.out.println("Save successfully");
            //weka.core.SerializationHelper.write("FFNNKF.model", FFNNKF);
        }*/
        Instance in = createInstanceFromInputUser(classifierFT.getInstances(),datasetFT);

        for (int j = 0; j < classifierFT.getInstances().numAttributes(); j++) {
            //Normalization
            if (classifierFT.getInstances().classIndex() != j) {
                updateValue = in.value(j) / classifierFT.getRange(j);
                in.setValue(j, updateValue);
            }
        }
        //System.out.println(neural1);

        double x[] = classifierFT.distributionForInstance(in);
        for(int k=0; k < classifierFT.getInstances().numClasses(); k++){
            System.out.println(x[k]);
        }

        int res = (int) classifierFT.classifyInstance(in);

        System.out.println("Result: " + classifierFT.getInstances().attribute(classifierFT.getInstances().classIndex()).value(res));

    }
}
