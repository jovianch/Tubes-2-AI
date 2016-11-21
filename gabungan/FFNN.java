import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.*;
import weka.filters.supervised.attribute.Discretize;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by raudi & joshua on 11/15/16.
 */
public class FFNN extends AbstractClassifier implements  OptionHandler, WeightedInstancesHandler, Randomizable, java.io.Serializable {

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
    public Instances instances;
    private double[] range;
    private int hiddenLayerNeuron = 5;

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        this.instances = instances;

        isNormalizeAttributes = true;
        double updateValue;

        //Normalize
        if (isNormalizeAttributes) {
            range = new double[this.instances.numAttributes()];
            for (int i=0; i<this.instances.numAttributes(); i++) {
                if (this.instances.classIndex() != i) {
                    if (this.instances.attribute(i).isNumeric()) {
                        range[i] = this.instances.kthSmallestValue(i, this.instances.size());
                    } else {
                        range[i] = this.instances.attribute(i).numValues();
                    }

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
                    }
                }
            }

        }

        edges = new ArrayList<Edge>();

        this.inputNeurons = new Neuron[this.instances.numAttributes()-1];
        this.hiddenNeurons = new Neuron[this.hiddenLayerNeuron];
        if (this.instances.numClasses() <= 2) {
            this.outputNeurons = new  Neuron[1];
        } else {
            this.outputNeurons = new  Neuron[this.instances.numClasses()];
        }

        for (int j = 0; j < this.hiddenLayerNeuron ; j++) {
            this.hiddenNeurons[j] = new Neuron();
        }

        for (int j = 0; j < this.outputNeurons.length; j++) {
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
        } else {
            for (int i = 0; i < this.instances.numAttributes() - 1; i++) {
                this.inputNeurons[i] = new Neuron();
                for (int j = 0; j < this.hiddenLayerNeuron ; j++) {
                    dummy = new Edge(this.inputNeurons[i], Math.random(), this.hiddenNeurons[j]);
                    this.edges.add(dummy);
                }
            }

            for (int i = 0; i <this.hiddenLayerNeuron; i++) {
                for (int j = 0; j < this.outputNeurons.length; j++) {
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
                for (int j = 0; j < this.instances.numAttributes(); j++) {
                    if (this.instances.classIndex() < j) {
                        this.inputNeurons[j-1].setOutputInput(this.instances.instance(i).value(j));
                    } else if (this.instances.classIndex() > j) {
                        this.inputNeurons[j].setOutputInput(this.instances.instance(i).value(j));
                    }
                }

                //OUTPUT HIDDEN LAYER
                for (int j = 0; j < this.hiddenLayerNeuron; j++) {
                    inputEdgeHidden = listEdge.getListTujuan(this.hiddenNeurons[j]);
                    this.hiddenNeurons[j].setOutput(inputEdgeHidden);
                }

                //OUTPUT OUTPUT LAYER
                for (int j = 0; j < this.outputNeurons.length; j++) {
                    inputEdgeHidden = listEdge.getListTujuan(this.outputNeurons[j]);
                    this.outputNeurons[j].setOutput(inputEdgeHidden);

                    //ERROR OUTPUT LAYER
                    if (this.instances.instance(i).value(this.instances.classIndex()) == j) {
                        this.outputNeurons[j].setErrorOutput(1);
                        sumerr = sumerr + (Math.pow(1 - this.outputNeurons[j].getOutput(),2) / this.instances.numClasses());
                    } else {
                        this.outputNeurons[j].setErrorOutput(0);
                        sumerr = sumerr + (Math.pow(0 - this.outputNeurons[j].getOutput(),2) / this.instances.numClasses());
                    }
                }

                for (int j = 0; j < this.hiddenLayerNeuron; j++) {
                    //ERROR HIDDEN LAYER
                    inputEdgeHidden = listEdge.getListSumber(this.hiddenNeurons[j]);
                    this.hiddenNeurons[j].setErrorHidden(inputEdgeHidden);
                }

                //UPDATE WEIGHT
                for (int j = 0; j < listEdge.getSize(); j++) {
                    double error = listEdge.getList().get(j).getTujuan().getError();
                    double input = listEdge.getList().get(j).getSumber().getOutput();
                    listEdge.getList().get(j).updateWeight(learningRate, error, input);
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
        List<Edge> inputEdgeHidden = null;

        StringBuffer model = new StringBuffer(10000);

        model.append("\n=== Classifier model (full training set) ===\n\n");
        for (int i=0; i<this.outputNeurons.length; i++) {
            model.append("Sigmoid Node " + i + "\n");
            model.append("    Inputs      Weights\n");
            inputEdgeHidden = this.listEdge.getListTujuan(this.outputNeurons[i]);

            for (int j=0; j<inputEdgeHidden.size(); j++) {
                model.append("    node "+(j+this.outputNeurons.length)+"      "+inputEdgeHidden.get(j).getWeight()+"\n");
            }
        }

        for (int i=0; i<this.hiddenNeurons.length; i++) {
            model.append("Sigmoid Node " + (i+this.outputNeurons.length) + "\n");
            model.append("    Inputs      Weights\n");
            inputEdgeHidden = this.listEdge.getListTujuan(this.hiddenNeurons[i]);

            for (int j=0; j<inputEdgeHidden.size(); j++) {
                model.append("    Attrib "+(this.instances.attribute(j).name())+"      "+inputEdgeHidden.get(j).getWeight()+"\n");
            }
        }

        for (int i=0; i<this.instances.numClasses(); i++) {
            model.append("Class " + (this.instances.attribute(this.instances.classIndex()).value(i)) + "\n");
            model.append("    Input\n");
            model.append("    Node " + i + "\n");
        }
        return model.toString();
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] result = new double[this.instances.numClasses()];

        for (int j = 0; j < this.instances.numAttributes() - 1; j++) {
            if (this.instances.classIndex() < j) {
                this.inputNeurons[j-1].setOutputInput(instance.value(j));
            } else if (this.instances.classIndex() > j) {
                this.inputNeurons[j].setOutputInput(instance.value(j));
            }
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
        for (int j = 0; j < this.outputNeurons.length; j++) {
            inputEdgeHidden = null;
            inputEdgeHidden = this.listEdge.getListTujuan(this.outputNeurons[j]);
            this.outputNeurons[j].setOutput(inputEdgeHidden);
            result[j] = this.outputNeurons[j].getOutput();
            sum += this.outputNeurons[j].getOutput();
        }

        if (this.instances.numClasses() <= 2) {
            if (result[0] < 0.5) {
                result[1] = 1;
            } else {
                result[1] = 0;
            }
        } else {
            for (int i = 0; i<this.instances.numClasses(); i++) {
                result[i] /= sum;
            }
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

