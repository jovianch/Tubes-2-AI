import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.neural.NeuralNode;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

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
    private listOfEdge listEdge = null;
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
    private Instances instances;
    private Filter filNorm;
    private double[] range;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        //Kayanya ga butuh discretize da, soalnya kita butuhnya numeric, dan data itu kalo ga numeric pasti nominal
        /*discretize = new Discretize();
        discretize.setInputFormat(instances);
        Instances discInstances = Filter.useFilter(instances, discretize);
        discInstances.setClassIndex(0);*/
        this.instances = instances;

        System.out.println(1 / (1 + (Math.exp(-0.2))));
        //Change nominal to numeric PR!!!!!
        for(int i=0;i<this.instances.numAttributes();i++) {
            if (this.instances.classIndex() != i) {
                if (this.instances.attribute(i).isNominal()) {
                    //Ubah ke numeric (belum nemu caranya)
                }
            }
        }

        isNormalizeAttributes = true;
        //Normalize kayanya bikin sendiri
        if (isNormalizeAttributes) {
            range = new double[this.instances.numAttributes()];
            for (int i=0; i<this.instances.numAttributes(); i++) {
                if (this.instances.classIndex() != i) {
                    range[i] = this.instances.kthSmallestValue(i, this.instances.size());
                    //System.out.println(range[i]);
                } else {
                    range[i] = 0;
                }
            }
            //Ubah valuenya berdasar range
            for (int i=0; i<this.instances.size(); i++) {
                for (int j=0; j<this.instances.numAttributes(); j++) {
                    if (this.instances.classIndex() != j) {
                        double updateValue = this.instances.instance(i).value(j) / this.range[j];
                        this.instances.instance(i).setValue(j, updateValue);
                        //System.out.println(this.instances.instance(i).value(j));
                    }
                }
            }

        }

        edges = new ArrayList<Edge>();

        this.inputNeurons = new Neuron[this.instances.numAttributes()-1];
        this.hiddenNeurons = new Neuron[12];
        this.outputNeurons = new  Neuron[this.instances.numClasses()];

        for (int j = 0; j < 12 ; j++) {
            this.hiddenNeurons[j] = new Neuron();
        }

        for (int j = 0; j < this.instances.numClasses(); j++) {
            this.outputNeurons[j] = new Neuron();
        }

        epoch = 100000;
        Edge dummy = null;
        for (int i = 0; i < this.instances.numAttributes() - 1; i++) {
            this.inputNeurons[i] = new Neuron();
            for (int j = 0; j < 12 ; j++) {
                dummy = new Edge(this.inputNeurons[i], Math.random(), this.hiddenNeurons[j]);
                this.edges.add(dummy);
            }
        }

        for (int i = 0; i <12; i++) {
            for (int j = 0; j < this.instances.numClasses(); j++) {
                dummy = new Edge(this.hiddenNeurons[i], Math.random(), this.outputNeurons[j]);
                this.edges.add(dummy);
            }
        }
            
        for(int k=0; k<epoch; k++) {
            random = new Random();
            this.instances.randomize(random);

            //System.out.println("iterasi ke - " + (k+1));
            
            this.listEdge = new listOfEdge(this.edges);
            //System.out.println("Jumlah atribut = " + this.instances.numAttributes());
            for (int i = 0; i < this.instances.size(); i++) {
                //System.out.println("Instance ke - " + (i + 1));
                
                for (int j = 0; j < this.instances.numAttributes() - 1; j++) {
                    //System.out.println("value ke - " + (j+1));
                    this.inputNeurons[j].setOutputInput(this.instances.instance(i).value(j));
                    //System.out.println(this.instances.instance(i).value(j));
                    //System.out.println(this.inputNeurons[j].getOutput());
                }

                //OUTPUT HIDDEN LAYER
                for (int j = 0; j < 12; j++) {
                    List<Edge> inputEdgeHidden = listEdge.getListTujuan(this.hiddenNeurons[j]);
                    //System.out.println(inputEdgeHidden.get(0).getWeight());
                    this.hiddenNeurons[j].setOutput(inputEdgeHidden);
                    //System.out.println("Output hiddenNeuron " + j + " : " + this.hiddenNeurons[j].getOutput());
                }

                //OUTPUT OUTPUT LAYER
                for (int j = 0; j < this.instances.numClasses(); j++) {
                    List<Edge> inputEdgeHidden = listEdge.getListTujuan(this.outputNeurons[j]);
                    //System.out.println(inputEdgeHidden.get(0).getWeight());
                    this.outputNeurons[j].setOutput(inputEdgeHidden);
                    //System.out.println("Output outputNeuron " + j + " : " + this.outputNeurons[j].getOutput());
                    
                    //ERROR OUTPUT LAYER
                    //double target = 0.5;
                    if (this.instances.instance(i).value(this.instances.classIndex()) == j) {
                        this.outputNeurons[j].setErrorOutput(1);
                        //System.out.println("Target 1");
                    } else {
                        this.outputNeurons[j].setErrorOutput(0);
                        //System.out.println("Target 0");
                    }
                    //System.out.println("Error outputNeuron " + j + " : " +  this.outputNeurons[j].getError());
                }

                for (int j = 0; j < 12; j++) {
                    //ERROR HIDDEN LAYER
                    List<Edge> inputEdgeHidden = listEdge.getListSumber(this.hiddenNeurons[j]);
                    this.hiddenNeurons[j].setErrorHidden(inputEdgeHidden);
                    //System.out.println("Error hiddenNeuron " + j + " : " + this.hiddenNeurons[j].getError());
                }

                //UPDATE WEIGHT
                for (int j = 0; j < listEdge.getSize(); j++) {
                    double error = listEdge.getList().get(j).getTujuan().getError();
                    double input = listEdge.getList().get(j).getSumber().getOutput();
                    //LEARNING RATE = 1.00
                    listEdge.getList().get(j).updateWeight(1.00, error, input);
                    //System.out.println("Weight " + j + "= " + listEdge.getList().get(j).getWeight());
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
            BufferedReader breader = new BufferedReader(new FileReader("Team.arff"));

            // Convert to Instances type
            instances = new Instances(breader);
            instances.setClassIndex(instances.numAttributes()-1);

            Classifier FFNN = new FFNN();

            FFNN.buildClassifier(instances);

            Evaluation eval = new Evaluation(instances);
            eval.evaluateModel(FFNN,instances);
            System.out.println("\nData Learning Using Full-Training Schema\n");
            System.out.println(eval.toString());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public String toString() {
        /*StringBuffer model;

        model = new StringBuffer(this.m_neuralNodes.length * 100);
        NeuralConnection[] var5 = this.m_neuralNodes;
        int var6 = var5.length;

        NeuralConnection[] inputs;
        int var7;
        int nob;
        for(var7 = 0; var7 < var6; ++var7) {
            NeuralConnection m_output = var5[var7];
            NeuralNode con = (NeuralNode)m_output;
            double[] weights = con.getWeights();
            inputs = con.getInputs();
            if(con.getMethod() instanceof SigmoidUnit) {
                model.append("Sigmoid ");
            } else if(con.getMethod() instanceof LinearUnit) {
                model.append("Linear ");
            }

            model.append("Node " + con.getId() + "\n    Inputs    Weights\n");
            model.append("    Threshold    " + weights[0] + "\n");

            for(nob = 1; nob < con.getNumInputs() + 1; ++nob) {
                if((inputs[nob - 1].getType() & 1) == 1) {
                    model.append("    Attrib " + this.m_instances.attribute(((MultilayerPerceptron.NeuralEnd)inputs[nob - 1]).getLink()).name() + "    " + weights[nob] + "\n");
                } else {
                    model.append("    Node " + inputs[nob - 1].getId() + "    " + weights[nob] + "\n");
                }
            }
        }

        MultilayerPerceptron.NeuralEnd[] var10 = this.m_outputs;
        var6 = var10.length;

        for(var7 = 0; var7 < var6; ++var7) {
            MultilayerPerceptron.NeuralEnd var11 = var10[var7];
            inputs = var11.getInputs();
            model.append("Class " + this.m_instances.classAttribute().value(var11.getLink()) + "\n    Input\n");

            for(nob = 0; nob < var11.getNumInputs(); ++nob) {
                if((inputs[nob].getType() & 1) == 1) {
                    model.append("    Attrib " + this.m_instances.attribute(((MultilayerPerceptron.NeuralEnd)inputs[nob]).getLink()).name() + "\n");
                } else {
                    model.append("    Node " + inputs[nob].getId() + "\n");
                }
            }
        }

        return model.toString();*/
        return "string";
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] result = new double[this.instances.numClasses()];
        //Change nominal to numeric
        /*for(int i=0;i<instance.numAttributes();i++) {
            if (this.instances.classIndex() != i) {
                if (instance.attribute(i).isNominal()) {
                    //Ubah ke numeric (belum nemu caranya)
                }
            }
        }

        //Normalize kayanya bikin sendiri
        if (isNormalizeAttributes) {
            //Ubah valuenya berdasar range
            for (int j=0; j<instance.numAttributes(); j++) {
                if (this.instances.classIndex() != j) {
                    double updateValue = instance.value(j) / this.range[j];
                    instance.setValue(j, updateValue);
                    System.out.println(instance.value(j));
                }
            }
        }*/

        for (int j = 0; j < instance.numAttributes() - 1; j++) {
            //System.out.println("value ke - " + (j+1));
            this.inputNeurons[j].setOutputInput(instance.value(j));
            //System.out.println(instance.value(j));
            //System.out.println(this.inputNeurons[j].getOutput());
        }

        List<Edge> inputEdgeHidden = null;

        //OUTPUT HIDDEN LAYER
        for (int j = 0; j < 12; j++) {
            inputEdgeHidden = null;
            inputEdgeHidden = this.listEdge.getListTujuan(this.hiddenNeurons[j]);
            //System.out.println(inputEdgeHidden.get(0).getWeight());
            this.hiddenNeurons[j].setOutput(inputEdgeHidden);
            //System.out.println(this.hiddenNeurons[j].getOutput());
        }

        //OUTPUT OUTPUT LAYER
        double maks = 0;
        int idxMaks = 0;
        for (int j = 0; j < this.instances.numClasses(); j++) {
            inputEdgeHidden = null;
            inputEdgeHidden = this.listEdge.getListTujuan(this.outputNeurons[j]);
            //System.out.println(inputEdgeHidden.get(0).getWeight());
            this.outputNeurons[j].setOutput(inputEdgeHidden);
            //System.out.println(this.outputNeurons[j].getOutput());
            result[j] = this.outputNeurons[j].getOutput();
            maks += this.outputNeurons[j].getOutput();
            /*if (this.outputNeurons[j].getOutput() > maks) {
                maks = this.outputNeurons[j].getOutput();
                idxMaks = j;
            }*/

            //ERROR OUTPUT LAYER
            //double target = 0.5;
            /*System.out.println("Ini class: " + instance.value(11));
            if (instance.value(this.instances.classIndex()) == j) {
                this.outputNeurons[j].setErrorOutput(1);
                System.out.println("Target 1");
            } else {
                this.outputNeurons[j].setErrorOutput(0);
                System.out.println("Target 0");
            }*/
            //System.out.println(this.outputNeurons[j].getError());
        }

        for (int i = 0; i<this.instances.numClasses(); i++) {
            result[i] /= maks;
            System.out.println("Hasil Distribusi " + i +" : " + result[i]);
        }

        /*for (int j = 0; j < 12; j++) {
            //ERROR HIDDEN LAYER
            inputEdgeHidden = null;
            inputEdgeHidden = listEdge.getListSumber(this.hiddenNeurons[j]);
            this.hiddenNeurons[j].setErrorHidden(inputEdgeHidden);
            //System.out.println(this.hiddenNeurons[j].getError());
        }*/

        //UPDATE WEIGHT
        /*for (int j = 0; j < listEdge.getSize(); j++) {
            double error = listEdge.getList().get(j).getTujuan().getError();
            double input = listEdge.getList().get(j).getSumber().getOutput();
            //LEARNING RATE = 1.00
            listEdge.getList().get(j).updateWeight(1.00, error, input);
            System.out.println("Weight " + j + "= " + listEdge.getList().get(j).getWeight());
        }*/

        return result;
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

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        return result;
    }
}
