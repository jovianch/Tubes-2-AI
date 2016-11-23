/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Joshua & Alif
 */

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Joshua & Alif
 */
public class NB extends AbstractClassifier {
    Instances dataset;
    double[][][] M;
    double[] M_idx;
    int index;              // letak index
    int n_index_value;      // banyaknya jenis nilai attribute index
    int n_attribute;        // banyaknya attribute dari dataset / instance
    int n_attribute_value;  // banyaknya jenis nilai pada suatu attribute
    
    @Override
    public void buildClassifier(Instances _dataset) throws Exception {
        index = 0;
        //Filtering jadi nominal semua
        Instances ntn_dataset = null;
        Instances disc_dataset = null;
        String[] options = new String[2];
        options[0]="-R";
        options[1]="first-last";
        
        String nama = "";
        int i;
        boolean found = false;
        for (i = 0;(i < _dataset.numAttributes())&&(!found);i++) {
            nama = _dataset.attribute(i).name();
            if (nama.equals("class")) {
                index = i;
                found = true;
            }
        }
        if (i == _dataset.numAttributes()) {
            index = _dataset.numAttributes()-1;
        }
        /*NumericToNominal ntn = new NumericToNominal();
        ntn.setOptions(options);
        ntn.setInputFormat(_dataset);
        ntn_dataset = Filter.useFilter(_dataset, ntn);
        ntn_dataset.setClassIndex(ntn_dataset.numAttributes()-1);*/
       
        //Discretize filtering
        Discretize disc = new Discretize();
        disc.setOptions(options);
        disc.setInputFormat(_dataset);
        disc_dataset = Filter.useFilter(_dataset, disc);
        disc_dataset.setClassIndex(index);
        
        dataset = disc_dataset;
        
        //Inisialisasi matriks M buat simpen probabilitas
        n_index_value = dataset.numClasses();
        n_attribute = dataset.numAttributes();
        //cari jenis attribute paling banyak
        int temp = -9999;
        for(i = 0; i<n_attribute; i++){
            if(dataset.attribute(i).numValues()>temp && i!=index){
                temp = dataset.attribute(i).numValues();
            }
        }
        n_attribute_value = temp;
        M = new double[n_attribute][n_attribute_value][n_index_value];
        for(i = 0; i<n_attribute; i++){
            for(int j=0; j<n_attribute_value; j++){
                for(int k=0; k<n_index_value; k++){
                    M[i][j][k] = 0;
                }
            }
        }
        
        M_idx = new double[n_index_value];
        for(i = 0; i<n_index_value; i++){
            M_idx[i] = 0;
        }
        //pembuatan matriks probabilitas
        Enumeration enu = dataset.enumerateInstances();
        while (enu.hasMoreElements()) {
            Instance _instance = (Instance) enu.nextElement();
            int n = _instance.numAttributes();
            for(int j=0; j<n ; j++){
                if(j!=index){
                    M[j][(int) _instance.value(j)][(int) _instance.value(index)]++;
                }
                else{
                    M_idx[(int) _instance.value(index)]++;
                }
            }
        }
        
        for(i = 0; i<n_attribute; i++){
            for(int k=0; k<n_index_value; k++){
                int total = 0;
                for(int j=0; j<n_attribute_value; j++){
                    total += M[i][j][k];
                }
                for(int j=0; j<n_attribute_value; j++){
                    M[i][j][k] = M[i][j][k] / total;   
                }
            }
        }
        int total = 0;
        for(i = 0; i<n_index_value; i++){
            total += M_idx[i];
        }
        for(i = 0; i<n_index_value; i++){
            M_idx[i] = M_idx[i] / total;
        }
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        double[] resultArr = new double[n_index_value];
        for(int k=0; k<n_index_value; k++){
            resultArr[k]=M_idx[k];
        }
        for(int k=0; k<n_index_value; k++){
            for(int i=0; i<n_attribute; i++){
                if(i!=index){
                    int x1 = (int) instnc.value(instnc.attribute(i));
                    resultArr[k] = resultArr[k] * M[i][x1][k];
                }
            }
        }
        return resultArr;
    }

     @Override
    public Capabilities getCapabilities() {
      Capabilities result = super.getCapabilities();
      result.disableAll();

      // attributes
      result.enable(Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capability.NUMERIC_ATTRIBUTES);
      result.enable( Capability.MISSING_VALUES );

      // class
      result.enable(Capability.NOMINAL_CLASS);
      result.enable(Capability.MISSING_CLASS_VALUES);

      // instances
      result.setMinimumNumberInstances(0);

      return result;
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double dist[] = new double[n_index_value];
        dist = distributionForInstance(instnc);
        double maxi = 0;
        double maxidx = 0;
        for (int i = 0;i < n_index_value;i++) {            
            if (maxi < dist[i]) {
                maxi = dist[i];
                maxidx = i;
            }
        }        
        return maxidx;
    }    
    
    @Override
    public String toString() {
        String output = "";
        for(int i = 0; i < n_attribute; i++){
            if(i != index){
                output = output + dataset.attribute(i).name() + "\n";
                for(int j = 0;j < dataset.attribute(i).numValues(); j++){
                    for(int k = 0;k < n_index_value; k++){
                            output = output + dataset.attribute(i).value(j)+ "(" + dataset.attribute(i).name() + ")" + " - " + dataset.attribute(index).value(k) + " : " + M[i][j][k] + "\n";
                    }
                }
            output = output + "\n\n";
            }
            else{
                for(int k = 0;k < n_index_value; k++){
                    output = output + dataset.attribute(i).value(k)+ " : " + M_idx[k] + "\n";
                }
            }
        }
        System.out.println("test " + output);
        return output;
    }
}
