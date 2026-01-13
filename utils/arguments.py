import argparse
import os

def get_arguments():
    
    parser = argparse.ArgumentParser(description="MLP Training Script")
    
    #ckpt
    parser.add_argument('--split_valid_by_paragraph', type=bool, default=False, help='Split validation by paragraph')
    
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint name for saving and loading the model')
    
    parser.add_argument('--use_bpr_loss', type=bool, default=False, help='Use BPR loss instead of BCEWithLogitsLoss')
    parser.add_argument('--bpr_loss_weight', type=float, default=0.25, help='Weight for BPR loss when using it')
    
    #================================================================#
    parser.add_argument('--is_kfold', type=bool, default=False, help='Use k-fold cross-validation') 
    parser.add_argument('--k_fold', type=int, default=10, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--fold_idx', type=int, default=0, help='Fold index for k-fold cross-validation')
    
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    
    
    #================= parser with data  ===========================#
    parser.add_argument('--max_length', type=int, default=256, help='Maximum length of input sequences')
    parser.add_argument('--use_paragraph', action='store_true', help='Use paragraph text for training')
    parser.add_argument('--add_title', action='store_true', help='Add title to the input sequences')
    parser.add_argument('--is_submission', type=bool, default=False, help='Whether to run in submission mode (no labels)')
    
    
    #================= parser with model  ===========================#    
    #logging_steps
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Per device eval batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--embedding_model', type=str, default='kykim/funnel-kor-base', help='Embedding model name or path')

    #================= parser with save, load  ===========================#
    parser.add_argument('--save_dir', type=str, default='baseline', help='Save directory') #save_name
    parser.add_argument('--save_name', type=str, default='test_logits', help='Name of the saved model file')
    parser.add_argument('--load_dir', type=str, default='baseline', help='Load directory')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for classification')

    #num_heads
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in the model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of attention heads in the model')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Number of attention heads in the model')
    parser.add_argument('--hidden_size', type=int, default=512, help='Number of attention heads in the model')
    parser.add_argument('--dropout', type=int, default=0., help='Number of attention heads in the model')
    
    # AIGT Model arguments (InfoNCE Loss)
    parser.add_argument('--use_infonce_loss', type=bool, default=False, 
                        help='Use InfoNCE Loss for contrastive learning')
    parser.add_argument('--lambda_cl', type=float, default=0.1, 
                        help='Weight for InfoNCE Loss')
    parser.add_argument('--temperature', type=float, default=0.07, 
                        help='Temperature for InfoNCE Loss')
    
    # Meta-Learning arguments
    parser.add_argument('--save_fold_logits', type=bool, default=False, 
                        help='Save fold logits for meta-learning')
    parser.add_argument('--meta_model_type', type=str, default='mlp', 
                        choices=['mlp', 'ridge'], 
                        help='Meta-classifier type')
    
    # Model name selection
    parser.add_argument('--model_name', type=str, default='AvsHModel',
                        choices=['AvsHModel', 'HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE'],
                        help='Model type to use')


    args = parser.parse_args()

    # Set local rank for distributed training
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    return args