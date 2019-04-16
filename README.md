# Names-classifier
Predicting names origins 

Comand for training:
allennlp train config.jsonnet -s serialization_dir --include-package names-classifier

Comand for predicting
allennlp predict ./serialization_dir/model.tar.gz ./data/test.txt --include-package names-classifier --predictor names-classifier
