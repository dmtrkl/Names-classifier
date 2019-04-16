// jsonnet allows local variables like this
local char_embedding_dim = 6;
local hidden_dim = 6;
local embedding_dim = char_embedding_dim;
local num_epochs = 10;
local patience = 10;
local batch_size = 1;


{
    "train_data_path": './data/train.txt',
    "validation_data_path": './data/val.txt',
    "dataset_reader": {
        "type": "names-reader",
        "token_indexers": {
            "token_characters": { "type": "characters" }
        }
    },
    "model": {
        "type": "names-model",
        "word_embeddings": {
            "token_embedders": {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": char_embedding_dim,
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": char_embedding_dim,
                        "hidden_size": char_embedding_dim
                    }
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
	"sorting_keys": [["name","num_token_characters"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "adam"
        },
        "patience": patience
    }
}


