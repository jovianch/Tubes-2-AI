import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.attribute.NominalToBinary;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by raudi on 11/15/16.
 */
public class FFNN extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler, Randomizable {

    private Neuron[] inputNeurons = null;
    private Neuron[] hiddenNeurons = null;
    private Neuron[] outputNeurons = null;

    private List<Edge> edges = null;

    private boolean isNumeric;
    private int numEpochs;
    private Random random;
    private boolean isUseNomToBin;
    private NominalToBinary ntb;
    private Discretize discretize;
    private boolean isNormalizeAttributes;
    private double learningRate;
    private double momentum;
    private int epoch = 0;
    private double error = 0.0D;
    private boolean isReset;
    private boolean isNormalizeClass;
    private int valSize;
    private int valThreshold;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        discretize = new Discretize();
        discretize.setInputFormat(instances);
        Instances discInstances = Filter.useFilter(instances, discretize);
        discInstances.setClassIndex(discInstances.numAttributes()-1);

        ntb = new NominalToBinary();
        ntb.setInputFormat(discInstances);
        Instances dataset = Filter.useFilter(discInstances, ntb);
        dataset.setClassIndex(discInstances.numAttributes()-1);
        edges = new ArrayList<Edge>();

        this.inputNeurons = new Neuron[dataset.numAttributes()-1];
        this.hiddenNeurons = new Neuron[2];
        this.outputNeurons = new  Neuron[dataset.numClasses()-1];
        epoch = 3;
        Edge dummy = null;
        for (int i = 0; i < dataset.numAttributes() - 1; i++) {
                this.inputNeurons[i] = new Neuron();
                for (int j = 0; j < 2; j++) {
                    this.hiddenNeurons[j] = new Neuron();
                    dummy = new Edge(this.inputNeurons[i], 1, this.hiddenNeurons[j]);
                    this.edges.add(dummy);
                }
            }

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < dataset.numClasses() - 1; j++) {
                    this.outputNeurons[j] = new Neuron();
                    dummy = new Edge(this.hiddenNeurons[i], 1, this.outputNeurons[j]);
                    this.edges.add(dummy);
                }
            }
            
        for(int k=0; k<epoch; k++) {
            System.out.println("iterasi ke - " + (k+1));
            
            listOfEdge listEdge = new listOfEdge(this.edges);
            System.out.println("Jumlah atribut = " + dataset.numAttributes());
            for (int i = 0; i < instances.size(); i++) {
                System.out.println("Instance ke - " + (i + 1));
                
                for (int j = 0; j < dataset.numAttributes() - 1; j++) {
                    System.out.println("value ke - " + (j+1));
                    this.inputNeurons[j].setOutputInput(instances.instance(i).value(j));
                }

                List<Edge> inputEdgeHidden = null;

                //OUTPUT HIDDEN LAYER
                for (int j = 0; j < 2; j++) {
                    inputEdgeHidden = null;
                    inputEdgeHidden = listEdge.getListTujuan(this.hiddenNeurons[j]);
                    //System.out.println(inputEdgeHidden.get(0).getWeight());
                    this.hiddenNeurons[j].setOutput(inputEdgeHidden);
                    //System.out.println(this.hiddenNeurons[j].getOutput());
                }

                //OUTPUT OUTPUT LAYER
                for (int j = 0; j < dataset.numClasses() - 1; j++) {
                    inputEdgeHidden = null;
                    inputEdgeHidden = listEdge.getListTujuan(this.outputNeurons[j]);
                    //System.out.println(inputEdgeHidden.get(0).getWeight());
                    this.outputNeurons[j].setOutput(inputEdgeHidden);
                    //System.out.println(this.outputNeurons[j].getOutput());
                    
                    //ERROR OUTPUT LAYER
                    //double target = 0.5;
                    this.outputNeurons[j].setErrorOutput(instances.instance(i).value(0));
                    //System.out.println(this.outputNeurons[j].getError());
                }

                for (int j = 0; j < 2; j++) {
                    //ERROR HIDDEN LAYER
                    inputEdgeHidden = null;
                    inputEdgeHidden = listEdge.getListSumber(this.hiddenNeurons[j]);
                    this.hiddenNeurons[j].setErrorHidden(inputEdgeHidden);
                    //System.out.println(this.hiddenNeurons[j].getError());
                }

                //UPDATE WEIGHT
                for (int j = 0; j < listEdge.getSize(); j++) {
                    double error = listEdge.getList().get(j).getTujuan().getError();
                    double input = listEdge.getList().get(j).getSumber().getOutput();
                    //LEARNING RATE = 1.00
                    listEdge.getList().get(j).updateWeight(1.00, error, input);
                    System.out.println("Weight " + j + "= " + listEdge.getList().get(j).getWeight());
                }

            }

        }

    }

    public void setOptions(String[] options) throws Exception {
        String learningString = Utils.getOption('L', options);
        if(learningString.length() != 0) {
            this.setLearningRate((new Double(learningString)).doubleValue());
        } else {
            this.setLearningRate(1);
        }

        String momentumString = Utils.getOption('M', options);
        if(momentumString.length() != 0) {
            this.setMomentum((new Double(momentumString)).doubleValue());
        } else {
            this.setMomentum(0);
        }

        String epochsString = Utils.getOption('N', options);
        if(epochsString.length() != 0) {
            this.setTrainingTime(Integer.parseInt(epochsString));
        } else {
            this.setTrainingTime(500);
        }

        String valSizeString = Utils.getOption('V', options);
        if(valSizeString.length() != 0) {
            this.setValidationSetSize(Integer.parseInt(valSizeString));
        } else {
            this.setValidationSetSize(0);
        }

        String seedString = Utils.getOption('S', options);
        if(seedString.length() != 0) {
            this.setSeed(Integer.parseInt(seedString));
        } else {
            this.setSeed(0);
        }

        String thresholdString = Utils.getOption('E', options);
        if(thresholdString.length() != 0) {
            this.setValidationThreshold(Integer.parseInt(thresholdString));
        } else {
            this.setValidationThreshold(20);
        }

        if(Utils.getFlag('B', options)) {
            this.setNominalToBinaryFilter(false);
        } else {
            this.setNominalToBinaryFilter(true);
        }

        if(Utils.getFlag('C', options)) {
            this.setNormalizeNumericClass(false);
        } else {
            this.setNormalizeNumericClass(true);
        }

        if(Utils.getFlag('I', options)) {
            this.setNormalizeAttributes(false);
        } else {
            this.setNormalizeAttributes(true);
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    public String[] getOptions() {
        Vector options = new Vector();
        options.add("-L");
        options.add("" + this.getLearningRate());
        options.add("-M");
        options.add("" + this.getMomentum());
        options.add("-N");
        options.add("" + this.getTrainingTime());
        options.add("-V");
        options.add("" + this.getValidationSetSize());
        options.add("-S");
        options.add("" + this.getSeed());
        options.add("-E");
        options.add("" + this.getValidationThreshold());

        if(!this.getNominalToBinaryFilter()) {
            options.add("-B");
        }

        if(!this.getNormalizeNumericClass()) {
            options.add("-C");
        }

        if(!this.getNormalizeAttributes()) {
            options.add("-I");
        }

        Collections.addAll(options, super.getOptions());
        return (String[])options.toArray(new String[0]);
    }

    public static void main(String[] args) {

        Instances instances = null;
        try {
            // Read file
            BufferedReader breader = new BufferedReader(new FileReader("mush.arff"));

            // Convert to Instances type
            instances = new Instances(breader);
            instances.setClassIndex(instances.numAttributes() - 1);

            Classifier FFNN = new FFNN();

            FFNN.buildClassifier(instances);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public void setSeed(int i) {

    }

    @Override
    public int getSeed() {
        return 0;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setTrainingTime(int trainingTime) {
        this.epoch = trainingTime;
    }

    public void setValidationSetSize(int validationSetSize) {
        this.valSize = validationSetSize;
    }

    public void setValidationThreshold(int validationThreshold) {
        this.valThreshold = validationThreshold;
    }

    public void setNominalToBinaryFilter(boolean nominalToBinaryFilter) {
        this.isUseNomToBin = nominalToBinaryFilter;
    }

    public void setNormalizeNumericClass(boolean normalizeNumericClass) {
        this.isNormalizeClass = normalizeNumericClass;
    }

    public void setNormalizeAttributes(boolean normalizeAttributes) {
        this.isNormalizeAttributes = normalizeAttributes;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public int getTrainingTime() {
        return epoch;
    }

    public int getValidationSetSize() {
        return valSize;
    }

    public int getValidationThreshold() {
        return valThreshold;
    }

    public boolean getNominalToBinaryFilter() {
        return isUseNomToBin;
    }

    public boolean getNormalizeNumericClass() {
        return isNormalizeClass;
    }

    public boolean getNormalizeAttributes() {
        return isNormalizeAttributes;
    }
}
