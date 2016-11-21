import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by raudi on 11/21/16.
 */
public class MainFFNN {
    /**
     * This method creates instance from input user
     *
     * @param numericData : dataset that have been fileterd by Discretize filter
     * @return instanceInput : instance that created by input user
     * @throws Exception
     */
    public static Instance createInstanceFromInputUser(Instances numericData, Instances plainData){

        boolean needNum = false;
        for (int i = 0;i < plainData.numAttributes();i++) {
            if (plainData.classIndex() != i) {
                needNum = needNum | plainData.attribute(i).isNominal();
            }
        }
        System.out.println(needNum);

        if (needNum) {
            // Initialize instance
            Instance instanceInput = new DenseInstance(numericData.numAttributes()+1);

            // Equate attribute types to the dataset's
            instanceInput.setDataset(plainData);

            // Read value of attributes from user input
            for(int i=0; i<plainData.numAttributes(); i++){
                if(i!=plainData.classIndex()){
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
        else {
            // Initialize instance
            Instance instanceInput = new DenseInstance(numericData.numAttributes()+1);

            // Equate attribute types to the dataset's
            instanceInput.setDataset(numericData);

            // Read value of attributes from user input
            for(int i=0; i<numericData.numAttributes(); i++){
                if(i!=numericData.classIndex()){
                    System.out.print("\nInput attribute " + instanceInput.attribute(i).name() + " : ");
                    Scanner s = new Scanner(System.in);
                    Float attributeValueInput = s.nextFloat();
                    instanceInput.setValue(i, attributeValueInput);
                }
            }

            // Print out the notification
            System.out.println("\n\nInstance have been created successfully by user input\n");
            return instanceInput;
        }
    }

    /**
     * This method does data learning by full-training schema
     *
     * @param numericDataset : dataset that have been filtered by Discretize filter
     * @return fullTrainingClassifier : classifier after learning
     */
    public static Classifier learningFullTraining(Instances numericDataset) {

        // Initialize classifier
        FFNN fullTrainingClassifier = new FFNN();
        try {
            // Initialize Evaluation
            Evaluation fullTrainingEvaluation = new Evaluation(numericDataset);

            // Build classifier and evaluation
            String[] options = new String[6];
            options[0]="-H";
            options[1]="22";
            options[2]="-L";
            options[3]="1";
            options[4]="-N";
            options[5]="100";
            fullTrainingClassifier.setOptions(options);
            fullTrainingClassifier.buildClassifier(numericDataset);
            fullTrainingEvaluation.evaluateModel(fullTrainingClassifier, numericDataset);

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
     * @param numericDataset : dataset that have been filtered by Discretize filter
     * @return fullTrainingClassifier : classifier after learning
     */
    public static Classifier learningKFoldCrossValidation(Instances numericDataset) {
        // Initialize classifier
        FFNN kFoldCrossValidationClassifier = new FFNN();
        try {
            // Initialize evaluation
            Evaluation kFoldCrossValidationEvaluation = new Evaluation(numericDataset);

            // Build classifier and evaluation
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;
            String[] options = new String[6];
            options[0]="-H";
            options[1]="22";
            options[2]="-L";
            options[3]="1";
            options[4]="-N";
            options[5]="100";
            kFoldCrossValidationClassifier.setOptions(options);

            kFoldCrossValidationEvaluation.crossValidateModel(kFoldCrossValidationClassifier, numericDataset, folds, rand);
            kFoldCrossValidationClassifier.buildClassifier(numericDataset);

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

    public static void main(String[] args) {

        Instances instances = null;
        try {
            boolean read = false;
            boolean save = false;
            Classifier FFNNKF = null;
            Classifier FFNNFT = null;
            if (read) {
                FFNNFT = (FFNN) weka.core.SerializationHelper.read("FFNN.model");
            } else {
                // Read file
                BufferedReader breader = new BufferedReader(new FileReader("Team.arff"));

                // Convert to Instances type
                instances = new Instances(breader);
                instances.setClassIndex(instances.numAttributes()-1);

                //FFNNKF = learningKFoldCrossValidation(instances);
                FFNNFT = learningFullTraining(instances);
            }

            Instances instancesPlain = new Instances(instances);
            instancesPlain.setClassIndex(instancesPlain.numAttributes()-1);
            Instance inputInstance = createInstanceFromInputUser(instances,instancesPlain);

            double x[] = FFNNFT.distributionForInstance(inputInstance);
            for(int k=0; k<instances.numClasses(); k++){
                System.out.println(x[k]);
            }

            if (save) {
                weka.core.SerializationHelper.write("FFNNFT.model", FFNNFT);
                weka.core.SerializationHelper.write("FFNNKF.model", FFNNKF);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
