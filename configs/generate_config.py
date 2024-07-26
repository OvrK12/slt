import yaml



def create_config(dataset_name, train_file, dev_file, test_file, 
                  learning_rate, batch_size, epochs, model_dir):
    config = {
        'name': dataset_name,
        'data': {
            'data_path': './data/',
            'version': 'phoenix_2014_trans',
            'sgn': 'sign',
            'txt': 'text',
            'gls': 'gloss',
            'train': train_file,
            'dev': dev_file,
            'test': test_file,
            'feature_size': 1024,
            'level': 'word',
            'txt_lowercase': True,
            'max_sent_length': 400,
            'random_train_subset': -1,
            'random_dev_subset': -1
        },
        'testing': {
            'recognition_beam_sizes': list(range(1, 11)),
            'translation_beam_sizes': list(range(1, 11)),
            'translation_beam_alphas': [-1, 0, 1, 2, 3, 4, 5]
        },
        'training': {
            'reset_best_ckpt': False,
            'reset_scheduler': False,
            'reset_optimizer': False,
            'random_seed': 42,
            'model_dir': model_dir,
            'recognition_loss_weight': 1.0,
            'translation_loss_weight': 1.0,
            'eval_metric': 'bleu',
            'optimizer': 'adam',
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_valid_log': 5,
            'epochs': epochs,
            'early_stopping_metric': 'eval_metric',
            'batch_type': 'sentence',
            'translation_normalization': 'batch',
            'eval_recognition_beam_size': 1,
            'eval_translation_beam_size': 1,
            'eval_translation_beam_alpha': -1,
            'overwrite': True,
            'shuffle': True,
            'use_cuda': True,
            'translation_max_output_length': 30,
            'keep_last_ckpts': 1,
            'batch_multiplier': 1,
            'logging_freq': 5,
            'validation_freq': 5,
            'betas': [0.9, 0.998],
            'scheduling': 'plateau',
            'learning_rate_min': 1.0e-07,
            'weight_decay': 0.001,
            'patience': 8,
            'decrease_factor': 0.7,
            'label_smoothing': 0.0
        },
        'model': {
            'initializer': 'xavier',
            'bias_initializer': 'zeros',
            'init_gain': 1.0,
            'embed_initializer': 'xavier',
            'embed_init_gain': 1.0,
            'tied_softmax': False,
            'encoder': {
                'type': 'transformer',
                'num_layers': 3,
                'num_heads': 8,
                'embeddings': {
                    'embedding_dim': 512,
                    'scale': False,
                    'dropout': 0.1,
                    'norm_type': 'batch',
                    'activation_type': 'softsign'
                },
                'hidden_size': 512,
                'ff_size': 2048,
                'dropout': 0.1
            },
            'decoder': {
                'type': 'transformer',
                'num_layers': 3,
                'num_heads': 8,
                'embeddings': {
                    'embedding_dim': 512,
                    'scale': False,
                    'dropout': 0.1,
                    'norm_type': 'batch',
                    'activation_type': 'softsign'
                },
                'hidden_size': 512,
                'ff_size': 2048,
                'dropout': 0.1
            }
        }
    }
    return config

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Add datasets for which you want to create configs here
datasets = [
    {
        'name': 'hands_mouth_whole',
        'train': 'bothHands+whole_data_train.pickle',
        'dev': 'bothHands+whole_data_dev.pickle',
        'test': 'bothHands+whole_data_test.pickle',
        'model_dir': './sign_sample_model/hands_mouth_whole'
    },
]

learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]
epochs_list = [10, 20]

for dataset in datasets:
    for lr in learning_rates:
        for bs in batch_sizes:
            for epochs in epochs_list:
                config = create_config(
                    dataset_name=dataset['name'],
                    train_file=dataset['train'],
                    dev_file=dataset['dev'],
                    test_file=dataset['test'],
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=epochs,
                    model_dir=dataset['model_dir']
                )
                
                filename = f"{dataset['name']}_lr{lr}_bs{bs}_epochs{epochs}.yaml"
                save_config(config, filename)
                print(f"Created config file: {filename}")