/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.util.*;

/**
 *
 * @author Jovian
 */
public class Neuron {
    private double output;
    private double error;
    
    public Neuron(){
        output = 0.00;
        error = 0.00;
    }
    
    public void setOutput(List<Edge> edge){
        double jumlah = (Math.random() * 0.1 - 0.1);
       
        //output = (1 / (1 + (Math.exp(-jumlah))));
        for (int i = 0; i < edge.size(); i++){
            jumlah = jumlah + (edge.get(i).getSumber().getOutput() * edge.get(i).getWeight());
        }
        output = (1 / (1 + (Math.exp(-jumlah))));
    }
    
    public void setOutputInput(double output){
        this.output = output;
    }
    
    public double getOutput(){
        return output;
    }
    
    public void setErrorOutput(double target){
        error = output * (1 - output) * (target - output);
    }
    
    public void setErrorHidden(List<Edge> edge){
        double sigma = 0;
        for (int i = 0; i < edge.size(); i++){
            sigma = sigma + (edge.get(i).getTujuan().getError() * edge.get(i).getWeight());
        }
        error = output * (1 - output) * sigma;
    }
    
    public double getError(){
        return error;
    }
    
}
