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
public class listOfEdge {
    private List<Edge> listEdge;
    
    public listOfEdge(List<Edge> listEdge){
        this.listEdge = listEdge;
    }
    
    public List<Edge> getListSumber(Neuron neuron){
        List<Edge> result = new ArrayList();
        for (int i = 0; i < listEdge.size(); i ++){
            if (listEdge.get(i).getSumber() == neuron){
                result.add(listEdge.get(i));
            }
        }
        return result;
    }
    
    public List<Edge> getListTujuan(Neuron neuron){
        List<Edge> result = new ArrayList();
        for (int i = 0; i < listEdge.size(); i ++){
            if (listEdge.get(i).getTujuan() == neuron){
                result.add(listEdge.get(i));
            }
        }
        return result;
    }
    
    public int getSize(){
        return listEdge.size();
    }
    
    public List<Edge> getList(){
        return listEdge;
    }
    
}
