import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Created by raudi on 11/15/16.
 */
public class FFNN extends AbstractClassifier implements  OptionHandler, WeightedInstancesHandler, Randomizable {

    private Neuron[] inputNeurons = null;
    private Neuron[] hiddenNeurons = null;
    private Neuron[] outputNeurons = null;

    private List<Edge> edges = null;
    private listOfEdge listEdge = null;
    private boolean isNumeric;
    private Random random;
    private Discretize discretize;
    private boolean isNormalizeAttributes;
    private double learningRate = 0.3;
    private int epoch = 1;
    private double error = 0.0D;
    private Instances instances;
    private double[] range;
    private int hiddenLayerNeuron = 5;

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        //Kayanya ga butuh discretize da, soalnya kita butuhnya numeric, dan data itu kalo ga numeric pasti nominal
        /*discretize = new Discretize();
        discretize.setInputFormat(instances);
        Instances discInstances = Filter.useFilter(instances, discretize);
        discInstances.setClassIndex(0);*/
        this.instances = instances;

        //Change nominal to numeric PR!!!!!
        /*for(int i=0;i<this.instances.numAttributes();i++) {
            if (this.instances.classIndex() != i) {
                if (this.instances.attribute(i).isNominal()) {
                    //Ubah ke numeric (belum nemu caranya)
                }
            }
        }*/

        isNormalizeAttributes = true;
        double updateValue;
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
            for (int i = 0; i < this.instances.size(); i++) {
                for (int j = 0; j < this.instances.numAttributes(); j++) {
                    //Normalization
                    if (this.instances.classIndex() != j) {
                        updateValue = this.instances.instance(i).value(j) / this.range[j];
                        this.instances.instance(i).setValue(j, updateValue);
                        //System.out.println(this.instances.instance(i).value(j));
                    }
                    //this.inputNeurons[j].setOutputInput(this.instances.instance(i).value(j));

                }
            }

        }

        edges = new ArrayList<Edge>();

        this.inputNeurons = new Neuron[this.instances.numAttributes()-1];
        this.hiddenNeurons = new Neuron[this.hiddenLayerNeuron];
        this.outputNeurons = new  Neuron[this.instances.numClasses()];

        for (int j = 0; j < this.hiddenLayerNeuron ; j++) {
            this.hiddenNeurons[j] = new Neuron();
        }

        for (int j = 0; j < this.instances.numClasses(); j++) {
            this.outputNeurons[j] = new Neuron();
        }

        Edge dummy = null;
        if (hiddenLayerNeuron == 0){ //SINGLE LAYER
             for (int i = 0; i < this.instances.numAttributes() - 1; i++) {
                this.inputNeurons[i] = new Neuron();
                for (int j = 0; j < this.instances.numClasses() ; j++) {
                    dummy = new Edge(this.inputNeurons[i], Math.random(), this.outputNeurons[j]);
                    this.edges.add(dummy);
                }
            }
        }
        else{ //MULTILAYER
            for (int i = 0; i < this.instances.numAttributes() - 1; i++) {
                this.inputNeurons[i] = new Neuron();
                for (int j = 0; j < this.hiddenLayerNeuron ; j++) {
                    dummy = new Edge(this.inputNeurons[i], Math.random(), this.hiddenNeurons[j]);
                    this.edges.add(dummy);
                }
            }

            for (int i = 0; i <this.hiddenLayerNeuron; i++) {
                for (int j = 0; j < this.instances.numClasses(); j++) {
                    dummy = new Edge(this.hiddenNeurons[i], Math.random(), this.outputNeurons[j]);
                    this.edges.add(dummy);
                }
            }
        }

        double sumerr = 999;
        double treshold = 0;
        random = new Random();
        List<Edge> inputEdgeHidden = null;
        for(int k=0; ((k<epoch) && (sumerr > treshold)); k++) {
            this.instances.randomize(random);

            System.out.println("iterasi ke - " + (k+1));
            sumerr = 0;

            this.listEdge = new listOfEdge(this.edges);
            for (int i = 0; i < this.instances.size(); i++) {
                for (int j = 0; j < this.instances.numAttributes() - 1; j++) {
                    //Normalization
                    /*if (this.instances.classIndex() != j) {
                        double updateValue = this.instances.instance(i).value(j) / this.range[j];
                        this.instances.instance(i).setValue(j, updateValue);
                        System.out.println(this.instances.instance(i).value(j));
                    }*/
                    this.inputNeurons[j].setOutputInput(this.instances.instance(i).value(j));

                }

                //OUTPUT HIDDEN LAYER
                for (int j = 0; j < this.hiddenLayerNeuron; j++) {
                    inputEdgeHidden = listEdge.getListTujuan(this.hiddenNeurons[j]);
                    //System.out.println(inputEdgeHidden.get(0).getWeight());
                    this.hiddenNeurons[j].setOutput(inputEdgeHidden);
                    //System.out.println("Output hiddenNeuron " + j + " : " + this.hiddenNeurons[j].getOutput());
                }

                //OUTPUT OUTPUT LAYER
                for (int j = 0; j < this.instances.numClasses(); j++) {
                    inputEdgeHidden = listEdge.getListTujuan(this.outputNeurons[j]);
                    //System.out.println(inputEdgeHidden.get(0).getWeight());
                    this.outputNeurons[j].setOutput(inputEdgeHidden);
                    //System.out.println("Output outputNeuron " + j + " : " + this.outputNeurons[j].getOutput());
                    
                    //ERROR OUTPUT LAYER
                    //double target = 0.5;
                    if (this.instances.instance(i).value(this.instances.classIndex()) == j) {
                        this.outputNeurons[j].setErrorOutput(1);
                        //System.out.println("Target 1");
                        sumerr = sumerr + (Math.pow(1 - this.outputNeurons[j].getOutput(),2) / this.instances.numClasses());
                    } else {
                        this.outputNeurons[j].setErrorOutput(0);
                        sumerr = sumerr + (Math.pow(0 - this.outputNeurons[j].getOutput(),2) / this.instances.numClasses());
                        //System.out.println("Target 0");
                    }
                    //System.out.println("Error outputNeuron " + j + " : " +  this.outputNeurons[j].getError());
                }

                for (int j = 0; j < this.hiddenLayerNeuron; j++) {
                    //ERROR HIDDEN LAYER
                    inputEdgeHidden = listEdge.getListSumber(this.hiddenNeurons[j]);
                    this.hiddenNeurons[j].setErrorHidden(inputEdgeHidden);
                    //System.out.println("Error hiddenNeuron " + j + " : " + this.hiddenNeurons[j].getError());
                }

                //UPDATE WEIGHT
                for (int j = 0; j < listEdge.getSize(); j++) {
                    double error = listEdge.getList().get(j).getTujuan().getError();
                    double input = listEdge.getList().get(j).getSumber().getOutput();
                    //LEARNING RATE = 1.00
                    listEdge.getList().get(j).updateWeight(learningRate, error, input);
                    //System.out.println("Weight " + j + "= " + listEdge.getList().get(j).getWeight());
                }
                

            }
            sumerr = sumerr / this.instances.numInstances();
            System.out.println("Error : " + sumerr);

        }

    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String learningString = Utils.getOption('L', options);
        if(learningString.length() != 0) {
            this.setLearningRate((new Double(learningString)).doubleValue());
        } else {
            this.setLearningRate(1);
        }

        String epochsString = Utils.getOption('N', options);
        if(epochsString.length() != 0) {
            this.setTrainingTime(Integer.parseInt(epochsString));
        } else {
            this.setTrainingTime(500);
        }

        String hiddenLayerNeuron = Utils.getOption('H', options);
        if(hiddenLayerNeuron.length() != 0) {
            this.setHiddenLayerNeuron(Integer.parseInt(hiddenLayerNeuron));
        } else {
            this.setHiddenLayerNeuron(5);
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
        options.add("-N");
        options.add("" + this.getTrainingTime());
        options.add("-H");
        options.add("" + this.getHiddenLayerNeuron());

        if(!this.getNormalizeAttributes()) {
            options.add("-I");
        }

        Collections.addAll(options, super.getOptions());
        return (String[])options.toArray(new String[0]);
    }

    public static void main(String[] args) {

        Instances instances = null;
        try {
            boolean read = false;
            boolean save = true;
            FFNN FFNN = null;
            if (read) {
                FFNN = (FFNN) weka.core.SerializationHelper.read("FFNN.model");
            } else {
                // Read file
                BufferedReader breader = new BufferedReader(new FileReader("Team.arff"));

                // Convert to Instances type
                instances = new Instances(breader);
                instances.setClassIndex(instances.numAttributes()-1);

                FFNN = new FFNN();
                String[] options = new String[6];
                options[0]="-H";
                options[1]="22";
                options[2]="-L";
                options[3]="1";
                options[4]="-N";
                options[5]="100";
                FFNN.setOptions(options);
                System.out.println(FFNN.getLearningRate());
                System.out.println(FFNN.getTrainingTime());
                System.out.println(FFNN.getHiddenLayerNeuron());
                FFNN.buildClassifier(instances);
            }

            Evaluation eval = new Evaluation(instances);
            eval.evaluateModel(FFNN,instances);
            System.out.println("\nData Learning Using Full-Training Schema\n");
            System.out.println(eval.toString());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

            if (save) {
                weka.core.SerializationHelper.write("FFNN.model", FFNN);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public String toString() {
        //StringBuffer model = new StringBuffer();


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

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] result = new double[this.instances.numClasses()];

        for (int j = 0; j < instance.numAttributes() - 2; j++) {
            this.inputNeurons[j].setOutputInput(instance.value(j));

        }

        List<Edge> inputEdgeHidden = null;

        //OUTPUT HIDDEN LAYER
        for (int j = 0; j < this.hiddenLayerNeuron; j++) {
            inputEdgeHidden = null;
            inputEdgeHidden = this.listEdge.getListTujuan(this.hiddenNeurons[j]);
            this.hiddenNeurons[j].setOutput(inputEdgeHidden);
        }

        //OUTPUT OUTPUT LAYER
        double sum = 0;
        for (int j = 0; j < this.instances.numClasses(); j++) {
            inputEdgeHidden = null;
            inputEdgeHidden = this.listEdge.getListTujuan(this.outputNeurons[j]);
            this.outputNeurons[j].setOutput(inputEdgeHidden);
            result[j] = this.outputNeurons[j].getOutput();
            sum += this.outputNeurons[j].getOutput();
        }

        for (int i = 0; i<this.instances.numClasses(); i++) {
            result[i] /= sum;
            //System.out.println("Hasil Distribusi " + i +" : " + result[i]);
        }

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

    public void setTrainingTime(int trainingTime) {
        this.epoch = trainingTime;
    }

    public void setNormalizeAttributes(boolean normalizeAttributes) {
        this.isNormalizeAttributes = normalizeAttributes;
    }

    public void setHiddenLayerNeuron(int n) {
        this.hiddenLayerNeuron = n;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getTrainingTime() {
        return epoch;
    }

    public int getHiddenLayerNeuron() {
        return this.hiddenLayerNeuron;
    }

    public double getRange(int i) {
        return this.range[i];
    }

    public Instances getInstances() {
        return this.instances;
    }

    public boolean getNormalizeAttributes() {
        return isNormalizeAttributes;
    }

    @Override
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

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double dist[] = new double[instances.numClasses()];
        dist = distributionForInstance(instnc);
        double maxi = 0;
        double maxidx = 0;
        for (int i = 0;i < instances.numClasses();i++) {
            if (maxi < dist[i]) {
                maxi = dist[i];
                maxidx = i;
            }
        }
        return maxidx;
    }
}
