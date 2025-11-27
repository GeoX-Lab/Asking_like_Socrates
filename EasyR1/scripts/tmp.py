from datasets import load_dataset

DATA_GLOB = "/g0001sr/lzy/RSVQA-HR_qwen_finetuning/data/train-*.parquet"

def main():
    ds = load_dataset("parquet", data_files={"train": DATA_GLOB}, split="train")
    print("总样本数:", len(ds))

if __name__ == "__main__":
    main()
