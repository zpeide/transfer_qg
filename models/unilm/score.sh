folder=$1
gold=$2
src=$3
python ../../metric/qgevalcap/eval.py -out "$folder/.validation" -src "$src" -tgt "$gold" > $1/score
cat $1/score
