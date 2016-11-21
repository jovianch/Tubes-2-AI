import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by raudi on 11/21/16.
 */
public class MainFFNN implements java.io.Serializable{
    /**
     * This method creates instance from input user
     *
     * @param numericData : dataset that have been fileterd by Discretize filter
     * @return instanceInput : instance that created by input user
     * @throws Exception
     */
    public static Instance createInstanceFromInputUser(Instances numericData, Instances plainData){

        boolean needDisc = false;
        for (int i = 0;i < plainData.numAttributes();i++) {
            needDisc = needDisc | plainData.attribute(i).isNumeric();
        }
        System.out.println(needDisc);

        if (needDisc) {
            // Initialize instance
            Instance instanceInput = new DenseInstance(numericData.numAttributes()+1);

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
            Instance instanceInput = new DenseInstance(numericData.numAttributes()+1);

            // Equate attribute types to the dataset's
            instanceInput.setDataset(numericData);

            // Read value of attributes from user input
            for(int i=0; i<numericData.numAttributes(); i++){
                if(i!=numericData.classIndex()){
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
    /**
     * This method does data learning by full-training schema
     *
     * @param numericDataset : dataset that have been filtered by Discretize filter
     * @return fullTrainingClassifier : classifier after learning
     */
    public static FFNN learningFullTraining(Instances numericDataset) {

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
    public static FFNN learningKFoldCrossValidation(Instances numericDataset) {
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
            options[5]="1000";
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
            boolean read = true;
            boolean save = true;
            FFNN FFNNKF = null;
            FFNN FFNNFT = null;
            
            // Read file
            BufferedReader breader = new BufferedReader(new FileReader("campuranTengah.arff"));

            // Convert to Instances type
            instances = new Instances(breader);
            instances.setClassIndex(10);

            //FFNNKF = learningKFoldCrossValidation(instances);
            FFNNKF = learningFullTraining(instances);
            //FFNNFT = learningFullTraining(instances);

            Instances instancesPlain = new Instances(instances);
            instancesPlain.setClassIndex(instances.classIndex());
            Instance inputInstance = createInstanceFromInputUser(instances,instancesPlain);

            double updateValue;
            for (int j = 0; j < FFNNKF.getInstances().numAttributes(); j++) {
                //Normalization
                if (FFNNKF.getInstances().classIndex() != j) {
                    updateValue = inputInstance.value(j) / FFNNKF.getRange(j);
                    inputInstance.setValue(j, updateValue);
                    //System.out.println(this.instances.instance(i).value(j));
                }
                //this.inputNeurons[j].setOutputInput(this.instances.instance(i).value(j));

            }


            if (save) {
         
                weka.core.SerializationHelper.write("FFNNKF.model", FFNNKF);
                System.out.println("Save successfully");
                //weka.core.SerializationHelper.write("FFNNKF.model", FFNNKF);
            }
            if (read) {
                FFNNKF = (FFNN) weka.core.SerializationHelper.read("FFNNKF.model");
                System.out.println(FFNNKF.toString());
            }
            

            double x[] = FFNNKF.distributionForInstance(inputInstance);
            for(int k=0; k<instances.numClasses(); k++){
                System.out.println(x[k]);
            }

            int res = (int) FFNNKF.classifyInstance(inputInstance);

            System.out.println("Result: " + instances.attribute(instances.classIndex()).value(res));
            //System.out.println(FFNNFT);

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
