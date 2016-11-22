import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 * Tugas Kecil 2 Artificial Intelligence
 * Exploration of Waikato Environment for Knowledge Analysis (WEKA)
 *
 * @author Alif Bhaskoro (13514016) and Praditya Raudi (13514087)
 */

public class TucilWeka {

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
			dataset.setClassIndex(dataset.numAttributes() - 1);

            // Print out the notification
            System.out.println("\nDataset have been read from iris.arff successfully\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dataset;
	}

    /**
     * This method applies filter in order to changes the dataset attribute
     *
     * @param dataset
     * @return discretizedDataset : dataset that have been filtered by Discretize filter
     */
	public static Instances filterDiscretize(Instances dataset) {

        // Initialize instances/dataset
		Instances discretizedDataset = null;
		try {
			// Set up option for filter -> default bin=10 and first until last attributes will be filtered
			String[] options = new String[2];
			options[0]="-R";
			options[1]="first-last";

			//  Build Discretize filter
			Discretize discretize = new Discretize();
			discretize.setOptions(options);
			discretize.setInputFormat(dataset);
			discretizedDataset = Filter.useFilter(dataset, discretize);
			discretizedDataset.setClassIndex(discretizedDataset.numAttributes()-1);

            // Print out the notification
            System.out.println("\nDataset have been filtered successfully\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return discretizedDataset;
	}

    /**
     * This method does data learning by 10-fold cross validation schema
     *
     * @param discretizedDataset : dataset that have been filtered by Discretize filter
	 * @return fullTrainingClassifier : classifier after learning
     */
	public static Classifier learningKFoldCrossValidation(Instances discretizedDataset) {

		// Initialize classifier
		Classifier kFoldCrossValidationClassifier = new J48();
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
     * This method does data learning by full-training schema
     *
     * @param discretizedDataset : dataset that have been filtered by Discretize filter
	 * @return fullTrainingClassifier : classifier after learning
     */
	public static Classifier learningFullTraining(Instances discretizedDataset) {

		// Initialize classifier
		Classifier fullTrainingClassifier = new J48();
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
     * This method saves/writes dataset model to external file
     *
     * @param filename : name and directory of external file
     * @param discretizedData : dataset that have been filtered by Discretize filter
     */
	public static void saveModel(Instances discretizedData, String filename) {

		try {
			// Initialize classifier
			Classifier classifier = new J48();
			classifier.buildClassifier(discretizedData);

			// Save model in external file
			weka.core.SerializationHelper.write(filename, classifier);

            // Print out the notification
            System.out.println("\nData model classifier have been saved successfully to '" + filename + "'\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

    /**
     * This method loads/reads dataset model from external file
     *
     * @param filename : name and directory of external file
     * @return classifier : dataset model classifier
     */
	public static Classifier readModel(String filename) {

		// Initialize classifier
		Classifier classifier = null;
		try {
            // Load model from external file
			classifier = (J48) weka.core.SerializationHelper.read(filename);

            // Print out the notification
            System.out.println("\nData model classifier have been loaded successfully from '" + filename + "'\n");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return classifier;
	}

    /**
     * This method creates instance from input user
     *
     * @param discretizedData : dataset that have been fileterd by Discretize filter
     * @return instanceInput : instance that created by input user
     * @throws Exception 
     */
	public static Instance createInstanceFromInputUser(Instances discretizedData){
		// Initialize instance
		Instance instanceInput = new DenseInstance(discretizedData.classIndex()+1);

        // Equate attribute types to the dataset's
		instanceInput.setDataset(discretizedData);

        // Read value of attributes from user input
		for(int i=0; i<discretizedData.classIndex(); i++){
			System.out.print("\nInput attribute " + instanceInput.attribute(i).name() + " : ");
			Scanner s = new Scanner(System.in);
			double attributeValueInput = s.nextDouble();
			instanceInput.setValue(i, attributeValueInput);
		}

		// Print out the notification
		System.out.println("\n\nInstance have been created successfully by user input\n");
		return instanceInput;
	}

    /**
     * This method classifies instance using certain classifier
     *
     * @param instanceInput : instance that created by input user
     * @param classifier : dataset model classifier to classify the instance
     */
	public static void classifyInstance(Classifier classifier, Instance instanceInput) {

        try {
            // Classify the instance
            double result = classifier.classifyInstance(instanceInput);
            String resultString = instanceInput.classAttribute().value((int) result);
            System.out.println(classifier.toString());

            // Print out the notification
            System.out.println("\nClassification result : " + resultString + "\n");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

	/**
	 * This method classifies instance using certain classifier
	 *
	 * @param instanceInput : instance that created by input user
	 * @param dataset : unfiltered dataset
	 * @param discretize : filter of dataset
	 * @return discretizeInstanceInput : instance that have been discretized
	 */
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

    public static void main(String[] argv) {

		int selection = 1;
		int schema = 0;
		Instance instanceInput = null;
		Instances dataset = null;
		Instances discretizedDataset = null;
		//  Build Discretize filter
		Discretize discretize = new Discretize();
		Classifier classifier = null;
		Instance discretizeInstanceInput = null;

		while (selection!=0) {
			System.out.println("=================================================");
			System.out.println("=================================================");
			System.out.println("=================================================");
			System.out.println("----------WELCOME TO TUGAS KECIL 2 WEKA----------");
			System.out.println("=================================================");
			System.out.println("=================================================");
			System.out.println("=================================================\n");

			System.out.println("                    MAIN MENU                    \n");
			System.out.println("                1. Input Dataset                 ");
			System.out.println("                2. Add Filter                    ");
			System.out.println("                3. Do learning Dataset            ");
			System.out.println("                4. Save Data Model               ");
			System.out.println("                5. Load Data Model               ");
			System.out.println("                6. Create Instance               ");
			System.out.println("                7. Classify           		     ");
			System.out.println("                0. Quit           		     \n");

			System.out.print("                Selection : ");
			Scanner s = new Scanner(System.in);
			selection = s.nextInt();

			switch (selection) {
				case 1 :
					// Membaca dataset yang diberikan
					dataset = readDataset("iris.arff");
					break;
				case 2 :
					if (dataset == null) {
						System.out.println("Input dataset first!\n");
					} else {
						// Mengaplikasikan filter yang mengubah tipe atribut, misalnya Discretize atau NumericToNominal.
						discretizedDataset = filterDiscretize(dataset);
					}
					break;
				case 3 :
					boolean repeat = true;
					while (repeat) {
						repeat = false;
						System.out.println("                 LEARNING DATASET                \n");
						System.out.println("                1. 10-Fold Cross Validation       ");
						System.out.println("                2. Full-Training                  ");
						System.out.println("                0. Back           				  ");
						System.out.print("                Selection : ");
						schema = s.nextInt();
						if (schema == 1) {
							if (discretizedDataset == null) {
								if (dataset == null) {
									System.out.println("Input dataset first!\n");
								} else {
									// Melakukan pembelajaran dataset dengan skema 10-fold cross validation
									classifier = learningKFoldCrossValidation(dataset);
								}
							} else {
								// Melakukan pembelajaran dataset dengan skema 10-fold cross validation
								classifier = learningKFoldCrossValidation(discretizedDataset);
							}
						} else if (schema == 2) {
							if (discretizedDataset == null) {
								if (dataset == null) {
									System.out.println("Input dataset first!\n");
								} else {
									// Melakukan pembelajaran dataset dengan skema full-training
									classifier = learningFullTraining(dataset);
								}
							} else {
								// Melakukan pembelajaran dataset dengan skema full-training
								classifier = learningFullTraining(discretizedDataset);
							}
						} else if (schema == 0) {
							break;
						} else {
							System.out.println("Input invalid\n");
							repeat = true;
						}
					}
					break;
				case 4 :
					if (discretizedDataset == null) {
						if (dataset == null) {
							System.out.println("Input dataset first!\n");
						} else {
							// Menyimpan (save) model/hipotesis hasil pembelajaran ke sebuah file eksternal
							saveModel(dataset,"iris.model");
						}
					} else {
						// Menyimpan (save) model/hipotesis hasil pembelajaran ke sebuah file eksternal
						saveModel(discretizedDataset,"iris.model");
					}
					break;
				case 5 :
					// Membaca (read) model/hipotesis dari file eksternal
					classifier = readModel("iris.model");
					break;
				case 6 :
					// Membuat instance baru sesuai masukan dari pengguna untuk setiap nilai atribut
					instanceInput = createInstanceFromInputUser(dataset);
					break;
				case 7 :
					if (discretizedDataset == null) {
						if (dataset == null) {
							System.out.println("Input dataset first!\n");
						} else {
							if (classifier == null) {
								System.out.println("Do learn dataset or load model first !\n");
							} else {
								if (instanceInput == null) {
									System.out.println("Create instance first!\n");
								} else {
									// Melakukan klasifikasi dengan memanfaatkan model/hipotesis dan instance sesuai masukan pengguna pada g.
									classifyInstance(classifier, instanceInput);
								}
							}
						}
					} else {
						if (classifier == null) {
							System.out.println("Do learn dataset or load model first !\n");
						} else {
							if (instanceInput == null) {
								System.out.println("Create instance first!\n");
							} else {
								discretizeInstanceInput = makeDiscretizeInstance(discretize,dataset,instanceInput);
								// Melakukan klasifikasi dengan memanfaatkan model/hipotesis dan instance sesuai masukan pengguna pada g.
								classifyInstance(classifier, discretizeInstanceInput);
							}
						}
					}

					break;
				case 0 :
					break;
				default :
					System.out.println("Input invalid\n");
					break;
			}
		}
	}
}
