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
public class Perceptron {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        Neuron neuron1 = new Neuron();
        Neuron neuron2 = new Neuron();
        Neuron neuron3 = new Neuron();
        Neuron neuron4 = new Neuron();
        Neuron neuron5 = new Neuron();
        List<Edge> edge = new ArrayList<Edge>();
        Edge dummy = new Edge(neuron1, 1.00, neuron3);
        edge.add(dummy);
        dummy = new Edge(neuron1, 1.00, neuron4);
        edge.add(dummy);
        dummy = new Edge(neuron2, 1.00, neuron3);
        edge.add(dummy);
        dummy = new Edge(neuron2, 1.00, neuron4);
        edge.add(dummy);
        dummy = new Edge(neuron3, 1.00, neuron5);
        edge.add(dummy);
        dummy = new Edge(neuron4, 1.00, neuron5);
        edge.add(dummy);
        
        listOfEdge listEdge = new listOfEdge(edge);
        //OUTPUT INPUT LAYER
        neuron1.setOutputInput(0.35);
        //System.out.println(neuron1.getOutput());
        neuron2.setOutputInput(0.9);
        
        //OUTPUT HIDDEN LAYER
        List<Edge> weight3 = new ArrayList<Edge>();
        weight3 = listEdge.getListTujuan(neuron3);
        System.out.println(weight3.get(0).getWeight());
        neuron3.setOutput(weight3);
        System.out.println(neuron3.getOutput());
        
        List<Edge> weight4 = new ArrayList<Edge>();
        weight4 = listEdge.getListTujuan(neuron4);
        neuron4.setOutput(weight4);
        System.out.println(neuron4.getOutput());
        
        //OUTPUT OUTPUT LAYER
        List<Edge> weight5 = new ArrayList<Edge>();
        weight5 = listEdge.getListTujuan(neuron5);
        neuron5.setOutput(weight5);
        System.out.println(neuron5.getOutput());
        
        //ERROR OUTPUT LAYER
        double target = 0.5;
        neuron5.setErrorOutput(target);
        System.out.println(neuron5.getError());
        
        //ERROR HIDDEN LAYER
        List<Edge> error4 = new ArrayList<Edge>();
        error4 = listEdge.getListSumber(neuron4);
        neuron4.setErrorHidden(error4);
        System.out.println(neuron4.getError());
        
        List<Edge> error3 = new ArrayList<Edge>();
        error3 = listEdge.getListSumber(neuron4);
        neuron3.setErrorHidden(error3);
        System.out.println(neuron3.getError());
        
        //UPDATE WEIGHT
        for (int i = 0; i < listEdge.getSize(); i++){
           double error = listEdge.getList().get(i).getTujuan().getError();
           double input = listEdge.getList().get(i).getSumber().getOutput();
           listEdge.getList().get(i).updateWeight(1.00, error, input);
           System.out.println("Weight " + i + "= " + listEdge.getList().get(i).getWeight());
        }
        
    }
    
}
