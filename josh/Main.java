
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
public class Main {
    private static int index = 0;
    /**
    * This method reads .arff file and convert it to dataset
    *
    * @param file : .arff file name and directory for making a dataset
    * @return dataset
    */
    public static Instances readDataset(String file) throws Exception {
 
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
            if (i == dataset.numAttributes()) {
                index = dataset.numAttributes()-1;
            }
            idxdel = index;
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

            // Print out the notification
            System.out.println("\nDataset have been read from iris.arff successfully\n");

            dataset = instNew;          
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
    public static Classifier learningFullTraining(Instances dataset, Instances dataset2) {
        Classifier fullTrainingClassifier = null;
        // Initialize classifier
        fullTrainingClassifier = new NB();
        NB sCls = new NB();
        try {
            // Initialize Evaluation
            Evaluation fullTrainingEvaluation = new Evaluation(dataset2);

            NB fullTrainingClassifier2 = new NB();

            // Build classifier and evaluation
            fullTrainingClassifier2.buildClassifier(dataset);
            //System.out.println(fullTrainingClassifier);
            sCls.buildClassifier(dataset2);

            fullTrainingEvaluation.evaluateModel(fullTrainingClassifier2, sCls.dataset);

            // Print out the result of classifier model and evaluation
            System.out.println("\nData Learning Using Full-Training Schema\n");
            //System.out.println(fullTrainingClassifier.toString());
            System.out.println(fullTrainingEvaluation.toSummaryString());
            System.out.println(fullTrainingEvaluation.toClassDetailsString());
            System.out.println(fullTrainingEvaluation.toMatrixString());
            return fullTrainingClassifier2;
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
    public static Classifier learningKFoldCrossValidation(Instances dataset, Instances dataset2) {
        Classifier kFoldClassifier = null;
        // Initialize classifier
        kFoldClassifier = new NB();
        NB sCls = new NB();
        try {
            // Initialize evaluation
            Evaluation kFoldCrossValidationEvaluation = new Evaluation(dataset2);

            // Build classifier and evaluation
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;
            kFoldClassifier.buildClassifier(dataset);
            //System.out.println(kFoldClassifier);
            sCls.buildClassifier(dataset2);
            //System.out.println(sCls.dataset);
            kFoldCrossValidationEvaluation.crossValidateModel(kFoldClassifier, sCls.dataset, folds, rand);                

            // Print out the result of classifier model and evaluation
            System.out.println("\nData Learning Using 10-Fold Cross Validation Schema\n");
            //System.out.println(kFoldClassifier.toString());
            System.out.println(kFoldCrossValidationEvaluation.toSummaryString());
            System.out.println(kFoldCrossValidationEvaluation.toClassDetailsString());
            System.out.println(kFoldCrossValidationEvaluation.toMatrixString());

        } catch (Exception e) {
                e.printStackTrace();
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
        boolean save = true, read = true, load = true;
        String filename = "";
        System.out.print("Enter filename: ");
        Scanner nama = new Scanner(System.in);
        filename = nama.next();

        NB bayes1 = null;

        if(load==false){
            Instances dataset = null;
            dataset = readDataset(filename);
            Instances preDisc = null;
            preDisc = readDataset(filename);
            
            bayes1 = new NB();
            bayes1.buildClassifier(dataset);
        }
        else{
            bayes1 = (NB) weka.core.SerializationHelper.read(filename);
        }
        Instances testdata = null;
        System.out.print("Enter test filename: ");        
        String filename2 = nama.next();        
        testdata = readDataset(filename2);
        
        Discretize discretize = new Discretize();
        
        Classifier kfold = learningKFoldCrossValidation(bayes1.dataset, bayes1.dataset);

        Classifier fullt = learningFullTraining(bayes1.dataset, testdata);
        
        bayes1.dataset.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(bayes1.dataset.numInstances() * 0.8);
        int testSize = bayes1.dataset.numInstances() - trainSize;
        Instances train = new Instances(bayes1.dataset, 0, trainSize);
        Instances test = new Instances(bayes1.dataset, trainSize, testSize);
        
        NB bayes2 = new NB();

        bayes2.buildClassifier(train);
        
        System.out.println("Split Test");
        Classifier splitCls = learningFullTraining(bayes2.dataset, test);
//            System.out.println(bayes1);
/*
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
  */
        weka.core.SerializationHelper.write("NBft.model", fullt);
        weka.core.SerializationHelper.write("NBkf.model", kfold);
        weka.core.SerializationHelper.write("NBst.model", splitCls);

            //NB bayes2 = new NB();
            //bayes2 = (NB) weka.core.SerializationHelper.read("NB.model");
            //print matriks nya
            //System.out.println(bayes2);
    }
}
