import json
import pandas as pd
import os
from sklearn.model_selection import KFold
import tempfile

from datasets.text_dataset import TextDataset
from datasets.text_collator import TextCollator

def get_dataset(args, tokenizer):
    """
    Get the dataset based on the provided arguments.
    
    Args:
        args: Configuration object containing dataset parameters.
        
    Returns:
        The dataset corresponding to the provided arguments.
    """

    if not args.is_submission:
        
        train_path = os.path.join(args.data_dir, "train.csv")
        
        if args.is_kfold:
            print("Using k-fold cross-validation")
            if args.k_fold <= 0:
                raise ValueError("k_fold must be a positive integer")

            train_df = pd.read_csv(train_path)
            
            constants_dir = None
            if 'baseline_paragraph' in args.save_dir:
                constants_dir = 'constants_phase2'
            elif 'baseline' in args.save_dir:
                constants_dir = 'constants_phase1'
            elif 'phase3' in args.save_dir:
                constants_dir = 'constants_phase3'
            else:
                constants_dir = 'constants_phase4'
            print(f"Using constants directory: {constants_dir}")
            
            os.makedirs(constants_dir, exist_ok=True)
            fold_path = os.path.join(constants_dir, "k_fold_split.json")

            # —————— 1) 파일 존재 여부 및 JSON 유효성 검사 ——————
            need_regenerate = False
            if os.path.exists(fold_path):
                try:
                    with open(fold_path, "r", encoding="utf-8") as f:
                        k_fold_split = json.load(f)
                    print("Loaded existing k_fold_split.json")
                except json.JSONDecodeError:
                    print("⚠️  k_fold_split.json is invalid JSON. Regenerating...")
                    need_regenerate = True
            else:
                need_regenerate = True

            # —————— 2) 필요시 새로 생성 ——————
            if need_regenerate:
                # raise NotImplementedError("K-Fold cross-validation is erased for consistency. "
                                        #   "Please implement it if needed.")
                kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
                k_fold_split = [
                    (train_idx.tolist(), val_idx.tolist())
                    for train_idx, val_idx in kf.split(train_df)
                ]

                # atomic write: 임시 파일에 쓰고 교체
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    delete=False,
                    dir=constants_dir,
                    encoding="utf-8"
                ) as tmp:
                    tmp_path = tmp.name
                    json.dump(k_fold_split, tmp, ensure_ascii=False, indent=2)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                os.replace(tmp_path, fold_path)
                print(f"✅  Saved new k-fold splits to {fold_path}")

            # —————— 3) Dataset 객체 생성 ——————
            fold_train = []
            fold_valid = []
            for train_idx, val_idx in k_fold_split:
                train_ds = TextDataset(
                    args,
                    train_df.iloc[train_idx],
                    tokenizer,
                    args.max_length,
                    is_train=True
                )
                valid_ds = TextDataset(
                    args,
                    train_df.iloc[val_idx],
                    tokenizer,
                    args.max_length,
                    is_train=False
                )
                fold_train.append(train_ds)
                fold_valid.append(valid_ds)

            print(f"Total folds: {len(fold_train)}")

            # 요청된 fold_idx 리턴
            return fold_train[args.fold_idx], fold_valid[args.fold_idx]

        else:
            print("Using standard train/validation split")
            print("Use FULLDATASET for training, not K-FOLD")
            print("validation is dummy, not real validation")    
            train_df = pd.read_csv(train_path)
            
            train_subset = train_df.sample(frac=1 - args.val_ratio, random_state=42)
            val_subset = train_df.drop(train_subset.index) 
            
            print(f"Train size: {len(train_df)}, Validation size: {len(val_subset)}")
            
            train_dataset = TextDataset(args, train_df, tokenizer, args.max_length, is_train = True)
            val_dataset = TextDataset(args, val_subset, tokenizer, args.max_length, is_train = False)
            return train_dataset, val_dataset

    else:
        test_path = os.path.join(args.data_dir, "test.csv")
        test_df = pd.read_csv(test_path)
        test_dataset = TextDataset(args, test_df, tokenizer, args.max_length, is_train = False, is_submission=True)
        print(f"Test size: {len(test_dataset)}")
                
        return None, test_dataset