import argparse
import os
import sys
from signjoey.training import train
from signjoey.prediction import test
from signjoey.fine_tune import fine_tune

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")

def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test", "fine_tune"], help="train a model, test, or fine-tune")
    
    ap.add_argument("config_path", type=str, help="path to YAML config file")
    
    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction or fine-tuning")
    
    ap.add_argument("--output_path", type=str, help="path for saving translation output")
    
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        if args.ckpt is None:
            raise ValueError("Checkpoint must be provided for testing")
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    elif args.mode == "fine_tune":
        if args.ckpt is None:
            raise ValueError("Checkpoint must be provided for fine-tuning")
        fine_tune(cfg_file=args.config_path, pre_trained_ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
