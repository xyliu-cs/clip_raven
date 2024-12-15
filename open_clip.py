import open_clip
import torch


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    state_dict = torch.load("PATH_TO_FINETUNED_PT")['state_dict']
    new_state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
