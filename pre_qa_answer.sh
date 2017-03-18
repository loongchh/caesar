cd core-nlp
echo "Starting nlp server"
nohup java -mx2g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer > /dev/null 2>/dev/null &
echo "Success!!"
cd ..
