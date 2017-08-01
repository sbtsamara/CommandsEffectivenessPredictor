package com.company;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.File;
import java.io.IOException;
import java.util.*;

//import ml.dmlc.xgboost4j.java.example.util.DataLoader;

public class Main {
    static void append(ArrayList<Float> X, ArrayList<Float> Y, ArrayList<Double> data) {
        for (double x : data) {
            X.add((float)x);
            Y.add(X.get(X.size() - 1));
        }
    }

    static ArrayList<String> getFiles(String dir){
        ArrayList<String> filesToLoad = new ArrayList<>();
        File dir1 = new File(dir);
        for (File f : dir1.listFiles()) {
            filesToLoad.add(f.toString());
        }
        Collections.sort(filesToLoad);
        return filesToLoad;
    }


    public static void main(String[] args) throws XGBoostError, IOException {
        HashMap<String, ArrayList<Double>> data = new HashMap<>(); // command -> list of percent

        ArrayList<String> filesToLoad = new ArrayList<>();

        filesToLoad.addAll(getFiles("/home/azaz/PycharmProjects/SBT/data/IFT results/april-may/"));
        filesToLoad.addAll(getFiles("/home/azaz/PycharmProjects/SBT/data/IFT results/may-june/"));

        for (String file : filesToLoad) {
            CsvOpener.appendMap(file, data);
        }

        Data data1 = new Data(data).invoke(0.3);
        ArrayList<Float> Xtrain = data1.getXtrain();
        ArrayList<Float> Ytrain = data1.getYtrain();
        ArrayList<Float> Xval = data1.getXval();
        ArrayList<Float> Yval = data1.getYval();
        ArrayList<Float> Xtest = data1.getXtest();
        ArrayList<Float> Ytest = data1.getYtest();


//        System.out.println(Xtrain);
//        System.out.println(Ytrain);
//        System.out.println(Xtrain.size());
//        System.out.println(Ytrain.size());
//
//        System.out.println(Xval);
//        System.out.println(Yval);
//        System.out.println(Xval.size());
//        System.out.println(Yval .size());
//
//        System.out.println(Xtest);
//        System.out.println(Ytest);
//        System.out.println(Xtest.size());
//        System.out.println(Ytest.size());

        DMatrix train = getdMatrix(Xtrain, Ytrain);
        DMatrix test = getdMatrix(Xtest, Ytest);

        Booster booster = getBooster(train,test);

        float[][] preds=booster.predict(test);

//        for (int j = 0; j < Xtrain.size(); j++) {
//            System.out.println(preds[j][0]+" "+Xtest.get(j));
//        }
    }

    private static DMatrix getdMatrix(ArrayList<Float> xtrain, ArrayList<Float> ytrain) throws XGBoostError {
        ArrayList<LabeledPoint> lp = new ArrayList<>();
        for(int i = 0; i< xtrain.size(); i++){
            LabeledPoint l=LabeledPoint.fromDenseVector(ytrain.get(i),new float[]{xtrain.get(i)});
            lp.add(l);
        }
        return new DMatrix(lp.iterator(),null);
    }

    private static Booster getBooster(DMatrix train,DMatrix test) throws XGBoostError {
        Map<String, Object> params = new HashMap<String, Object>();
        params.put("eta", 1f);
        params.put("max_depth", 15);
        params.put("subsample", 1f);
//        params.put("silent", 1);
//        params.put("objective", "reg:linear");


        HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
        watches.put("train", train);
        watches.put("test", test);

        int round = 30;
        Booster booster = XGBoost.train(train, params, 30, watches, null, null);
        booster.saveModel("model.dat");
        return booster;
    }

    private static class Data {
        private HashMap<String, ArrayList<Double>> data;
        private ArrayList<Float> xtrain;
        private ArrayList<Float> ytrain;
        private ArrayList<Float> xval;
        private ArrayList<Float> yval;
        private ArrayList<Float> xtest;
        private ArrayList<Float> ytest;

        public Data(HashMap<String, ArrayList<Double>> data) {
            this.data = data;
        }

        public ArrayList<Float> getXtrain() {
            return xtrain;
        }

        public ArrayList<Float> getYtrain() {
            return ytrain;
        }

        public ArrayList<Float> getXval() {
            return xval;
        }

        public ArrayList<Float> getYval() {
            return yval;
        }

        public ArrayList<Float> getXtest() {
            return xtest;
        }

        public ArrayList<Float> getYtest() {
            return ytest;
        }

        public Data invoke(double testSplit) {
            xtrain = new ArrayList<>();
            ytrain = new ArrayList<>();
            xval = new ArrayList<>();
            yval = new ArrayList<>();
            xtest = new ArrayList<>();
            ytest = new ArrayList<>();
            Random rand = new Random();

            int i = 0;
            xtrain.add(0.0f);
            xval.add(0.0f);
            xtest.add(0.0f);

            for (HashMap.Entry<String, ArrayList<Double>> entry : data.entrySet()) {
                if (i % 9 == 0) {
                    append(xval, yval, entry.getValue());
                } else {
                    if(rand.nextDouble()>testSplit) {
                        append(xtrain, ytrain, entry.getValue());
                    }else{
                        append(xtest, ytest, entry.getValue());
                    }
                }
                i += 1;
            }

            ytrain.add(xtrain.get(xtrain.size()-1));
            yval.add(xval.get(xval.size()-1));
            ytest.add(xtest.get(xtest.size()-1));
            return this;
        }
    }
}