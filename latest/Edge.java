/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Jovian & Joshua
 */
public class Edge implements java.io.Serializable{
    private Neuron sumber;
    private Neuron tujuan;
    private double weight;

    public Edge(){
        sumber = null;
        tujuan = null;
        weight = 0.00;
    }

    public Edge(Neuron sumber, double weight, Neuron tujuan){
        this.sumber = sumber;
        this.weight = weight;
        this.tujuan = tujuan;
    }

    public Neuron getSumber(){
        return sumber;
    }

    public Neuron getTujuan(){
        return tujuan;
    }

    public double getWeight(){
        return weight;
    }

    public void updateWeight(double n, double error, double input){
        weight = weight + n * error * input;
    }
    
    public void setWeight(double w) {
        this.weight = w;
    }

}