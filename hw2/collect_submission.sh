files="LSTM_Captioning.ipynb
NetworkVisualization.ipynb
RNN_Captioning.ipynb
StyleTransfer.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required notebook $file not found."
        exit 0
    fi
done


rm -f assignment2.zip
zip -q -r assignment2.zip . -x \
    "*.git" "*deeplearning/datasets*" "*.ipynb_checkpoints*" "*README.md" \
    "*collect_submission.sh" "*requirements*.txt" ".env/*" "*.pyc" \
    "*deeplearning/build/*" "*.pt" "*.npz"
