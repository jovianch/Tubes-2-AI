
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
    private static int choice = 0;
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
    public static Classifier learningFullTraining(Instances dataset) {
        Classifier fullTrainingClassifier = null;
        // Initialize classifier
        if (choice == 1){
            fullTrainingClassifier = new NB();
            try {
                // Initialize Evaluation
                Evaluation fullTrainingEvaluation = new Evaluation(dataset);

                // Build classifier and evaluation
                fullTrainingClassifier.buildClassifier(dataset);
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
        } else if (choice == 2) {
            fullTrainingClassifier = new FFNN();
            try {
                FFNN FTC = (FFNN) fullTrainingClassifier;
                // Initialize Evaluation
                Evaluation fullTrainingEvaluation = new Evaluation(dataset);

                // Build classifier and evaluation
                String[] options = new String[6];
                options[0]="-H";
                options[1]="22";
                options[2]="-L";
                options[3]="1";
                options[4]="-N";
                options[5]="100";
                FTC.setOptions(options);
                FTC.buildClassifier(dataset);
                fullTrainingEvaluation.evaluateModel(FTC, dataset);

                // Print out the result of classifier model and evaluation
                System.out.println("\nData Learning Using Full-Training Schema\n");
                System.out.println(FTC.toString());
                System.out.println(fullTrainingEvaluation.toSummaryString());
                System.out.println(fullTrainingEvaluation.toClassDetailsString());
                System.out.println(fullTrainingEvaluation.toMatrixString());

                fullTrainingClassifier = FTC;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return fullTrainingClassifier;
    }

    /**
     * This method does data learning by 10-fold cross validation schema
     *
     * @param discretizedDataset : dataset that have been filtered by Discretize filter
         * @return fullTrainingClassifier : classifier after learning
     */
    public static Classifier learningKFoldCrossValidation(Instances dataset) {
        Classifier kFoldClassifier = null;
        
        if (choice == 1) {
            // Initialize classifier
            kFoldClassifier = new NB();
            try {
                // Initialize evaluation
                Evaluation kFoldCrossValidationEvaluation = new Evaluation(dataset);

                // Build classifier and evaluation
                Random rand = new Random(1);  // using seed = 1
                int folds = 10;

                kFoldCrossValidationEvaluation.crossValidateModel(kFoldClassifier, dataset, folds, rand);
                kFoldClassifier.buildClassifier(dataset);

                // Print out the result of classifier model and evaluation
                System.out.println("\nData Learning Using 10-Fold Cross Validation Schema\n");
                System.out.println(kFoldClassifier.toString());
                System.out.println(kFoldCrossValidationEvaluation.toSummaryString());
                System.out.println(kFoldCrossValidationEvaluation.toClassDetailsString());
                System.out.println(kFoldCrossValidationEvaluation.toMatrixString());

            } catch (Exception e) {
                    e.printStackTrace();
            }
        } else if (choice == 2) {
            kFoldClassifier = new FFNN();
            FFNN KFC = (FFNN) kFoldClassifier;
            try {
                // Initialize evaluation
                Evaluation kFoldCrossValidationEvaluation = new Evaluation(dataset);

                // Build classifier and evaluation
                Random rand = new Random(1);  // using seed = 1
                int folds = 10;
                String[] options = new String[6];
                options[0]="-H";
                options[1]="7";
                options[2]="-L";
                options[3]="1";
                options[4]="-N";
                options[5]="100";
                KFC.setOptions(options);

                kFoldCrossValidationEvaluation.crossValidateModel(KFC, dataset, folds, rand);
                KFC.buildClassifier(dataset);

                // Print out the result of classifier model and evaluation
                System.out.println("\nData Learning Using 10-Fold Cross Validation Schema\n");
                System.out.println(KFC.toString());
                System.out.println(kFoldCrossValidationEvaluation.toSummaryString());
                System.out.println(kFoldCrossValidationEvaluation.toClassDetailsString());
                System.out.println(kFoldCrossValidationEvaluation.toMatrixString());

                kFoldClassifier = KFC;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return kFoldClassifier;
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
        
        while ((choice < 1) || (choice > 2)) {
            System.out.println("Enter (1) for Naive Bayes, (2) for Feedforward Neural Network:");
            choice = nama.nextInt();
        }                        
        
        if (choice == 1) {
            Discretize discretize = new Discretize();

            NB bayes1 = new NB();        

            bayes1.buildClassifier(dataset);

            Classifier kfold = learningKFoldCrossValidation(bayes1.dataset);
            Classifier fullt = learningFullTraining(bayes1.dataset);

            System.out.println(bayes1);

            Instance in = createInstanceFromInputUser(bayes1.dataset, preDisc);
            //System.out.println("The instance is " + in);
            Instance fi = makeDiscretizeInstance(discretize, dataset, in);
            double x[] = bayes1.distributionForInstance(fi);
            for(int k=0; k<bayes1.n_index_value; k++){
                System.out.println(bayes1.dataset.attribute(bayes1.index).value(k) + " - " + x[k]);
            }
            int res = (int) bayes1.classifyInstance(fi);
            System.out.println("Result: " + bayes1.dataset.attribute(bayes1.index).value(res));
            nama.next();
            weka.core.SerializationHelper.write("NB.model", bayes1);

            NB bayes2 = new NB();
            bayes2 = (NB) weka.core.SerializationHelper.read("NB.model");
            //print matriks nya
            System.out.println(bayes2);
        } else if (choice == 2) {
            FFNN neural1 = new FFNN();
            
            neural1.buildClassifier(dataset);
            
            FFNN fullt = new FFNN();
            
            Classifier kfold = learningKFoldCrossValidation(neural1.getInstances());
            fullt = (FFNN) learningFullTraining(neural1.getInstances());
            
            Instance in = createInstanceFromInputUser(neural1.getInstances(),preDisc);
            
            double updateValue;
            for (int j = 0; j < fullt.getInstances().numAttributes(); j++) {
                //Normalization
                if (fullt.getInstances().classIndex() != j) {
                    updateValue = in.value(j) / fullt.getRange(j);
                    in.setValue(j, updateValue);
                    //System.out.println(this.getInstances().instance(i).value(j));
                }
                //this.inputNeurons[j].setOutputInput(this.getInstances().instance(i).value(j));
            }
            System.out.println(neural1);
            
            double x[] = fullt.distributionForInstance(in);
            for(int k=0; k < fullt.getInstances().numClasses(); k++){
                System.out.println(x[k]);
            }

            int res = (int) fullt.classifyInstance(in);

            System.out.println("Result: " + fullt.getInstances().attribute(fullt.getInstances().classIndex()).value(res));
            //System.out.println(FFNNFT);
            nama.next();
            
            if (save) {
                weka.core.SerializationHelper.write("FFNNKF.model", fullt);
                System.out.println("Save successfully");
                //weka.core.SerializationHelper.write("FFNNKF.model", FFNNKF);
            }
            if (read) {
                FFNN neural2 = null;
                neural2 = (FFNN) weka.core.SerializationHelper.read("FFNNKF.model");
                System.out.println(neural2);
            }
        }        
                
        
    }
}
