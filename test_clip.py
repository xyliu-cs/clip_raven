import pandas as pd
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

file_path = "/apdcephfs_qy3/share_301812049/xiaoyuanliu/data/RAVEN/test_with_held_out_formatted.csv"
df = pd.read_csv(file_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/apdcephfs_qy3/share_301812049/xiaoyuanliu/Github/open_clip/src/logs/2024_12_08-23_44_12-model_ViT-L-14-lr_5e-06-b_64-j_4-p_amp_bf16/checkpoints/epoch_32.pt"
model_name = "ViT-L-14"

model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name, pretrained = model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)

correct_predictions = 0
total_predictions = 0
for i in tqdm(range(100)):
    img_path = df.iloc[i]["filepath"]
    ans = df.iloc[i]["caption"]
    image = preprocess(Image.open(img_path)).unsqueeze(0).cuda(device=device)
    options = ["1", "2", "3", "4", "5", "6"]
    text = tokenizer(options).cuda(device=device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_index = text_probs.argmax(dim=-1).item()
        # print(predicted_index)
        predicted_label = options[predicted_index]
        correct = (predicted_label == ans)
        if correct:
            correct_predictions += 1
        total_predictions += 1
        print(f"Predicted label: {predicted_label}, Actual label: {ans}, Correct: {correct}")

accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")