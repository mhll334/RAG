import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


model_name = "raynardj/classical-chinese-punctuation-guwen-biaodian"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

id2label = {
    0: "",
    1: "，",
    2: "。",
    3: "？",
    4: "！",
    5: "、"
}

def add_punctuation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    result = ""
    for token, label_id in zip(tokens[1:-1], predictions[1:-1]):
        token = token.replace("▁", "")
        result += token + id2label.get(label_id, "")
    return result

if __name__ == "__main__":
    input_folder = os.path.expanduser("/Users/macbook/Desktop/transcripts2")  
    output_folder = os.path.join(input_folder, "with_punctuation")  

    os.makedirs(output_folder, exist_ok=True)  

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            punctuated = add_punctuation(text)

            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(punctuated)

            print(f"處理完成：{filename}")
