
/**
 * Created by azaz on 25.07.17.
 */

import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.pipe.iterator.StringArrayIterator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Word2VecModel;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class Main {

    public static final File binFile = new File("/home/azaz/PycharmProjects/SBT/Models/ruscorpora_mean_hs.model.bin");
    public static Word2VecModel binModel;


    public static void main(String[] args) throws Exception {
        preprocessFile("filtered_logs_1.tsv", "stammed.txt");
        ParallelTopicModel model = trainModel(50,"stammed.txt");
        TestModel("models/model_Logs.bin.500");
    }

    private static void preprocessFile(String input, String output) throws IOException {

        PrintWriter pw = new PrintWriter(new FileWriter(new File(output)), false);

        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
        pipeList.add(new CharSequenceLowercase());
        pipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        pipeList.add(new TokenSequence2Stem());
        pipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));
        pipeList.add(new TokenSequence2File(pw));

        InstanceList instances = new InstanceList(new SerialPipes(pipeList));

        Pattern p = Pattern.compile("" +
                "([^\t]*\\t){6}" +
                "([^\t]*\\t)" +
                "(.*)");
        instances.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(input)),
                        p, 2, -1, -12
                )
        );
        pw.flush();
        pw.close();

    }

    private static void TestModel(String filename) throws Exception {
        ParallelTopicModel model = ParallelTopicModel.read(new File(filename));
        System.out.println("Loaded");
        Object[][] topWords = model.getTopWords(10);

        ArrayList<Pipe> pipeList = new ArrayList<>();
        pipeList.add(new CharSequenceLowercase());
        pipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        pipeList.add(new TokenSequence2Stem());
        pipeList.add(new TokenSequenceRemoveStopwords(new File("stopStem.txt"), "UTF-8", false, false, false));
        pipeList.add(new TokenSequence2FeatureSequence());

        InstanceList instances = new InstanceList(new SerialPipes(pipeList));


        instances.addThruPipe(
                new StringArrayIterator(
                        new String[]{
                                "при сохранении документа это указывается как обязательное поле",
                                "С данным заявление только в банк по месту обслуживания. Служба технической поддержки не принимает и не обрабатывает электронные документы,письма и платежные поручения."
                        }

                )
        );
        TopicInferencer inferencer = model.getInferencer();
        for (Instance instance : instances) {
            double[] sampledDistribution = inferencer.getSampledDistribution(instances.get(0), 50, 1, 5);
            TreeMap<Double, Integer> probs = new TreeMap<>();
            int[] k = {0};

            Arrays.stream(sampledDistribution).forEach(v -> probs.put(v, k[0]++));
            int i = 0;
            for (Map.Entry<Double, Integer> e : probs.descendingMap().entrySet()) {
                System.out.println(e.getKey() + " " + Arrays.toString(topWords[e.getValue()]));
                i++;
                if (i == 5) {
                    break;
                }
            }
            System.out.println("============================================");
        }
    }

    public static ParallelTopicModel trainModel(int topicCount,String filename) throws IOException {
        ParallelTopicModel model = new ParallelTopicModel(topicCount);

        InstanceList pipeline;
        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
        pipeList.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));
        pipeList.add(new TokenSequence2FeatureSequence());
        pipeline = new InstanceList(new SerialPipes(pipeList));
        Pattern p = Pattern.compile("(.*)");
        pipeline.addThruPipe(
                new CsvIterator(
                        new FileReader(new File(filename)),
                        p, 1, -1, -1
                )
        );

        model.addInstances(pipeline);

        model.setNumThreads(4);
        model.setNumIterations(500);
        model.setSaveSerializedModel(50, "./models/model_Logs.bin");
        model.estimate();

        return model;
    }

    private static List<Double> getVector(String s) {
        try {
            System.out.println(binModel.forSearch().getMatches(s, 10));
            return binModel.forSearch().getRawVector(s);

        } catch (Searcher.UnknownWordException e) {
            e.printStackTrace();
        }
        return null;


    }
}
