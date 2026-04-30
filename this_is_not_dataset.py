from datasets import load_dataset

dataset = load_dataset("HiTZ/This-is-not-a-dataset", streaming=True, split="test")

for i, example in enumerate(dataset):
    print(example)
    if i == 10:
        break
