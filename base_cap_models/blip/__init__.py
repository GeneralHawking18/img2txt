from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import torch.nn as nn

class BLIP_1(nn.Module):
    def __init__(self, mode = "eval"):
        super().__init__()
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name_model = "Salesforce/blip-image-captioning-base"
        self.processors = AutoProcessor.from_pretrained(name_model)
        self.model = BlipForConditionalGeneration.from_pretrained(name_model)
        self.init_prompt = 'A photo of'

    def set_init_prompt(self, prompt):
        self.init_prompt = prompt
    
    def get_transforms(self):
        return self.processors

    def forward(self, image):
        inputs = self.processors(images=image, text = self.init_prompt, return_tensors="pt")
        outs = self.model.generate(**inputs)
        captions = self.processors.decode(outs[0], skip_special_tokens=True)
        return self.remove_redundances(captions)
    
    def remove_redundances(self, captions):
        if isinstance(captions, list):
            for i in range(len(captions)):
                captions[i] = captions[i].replace(self.init_prompt.lower() + ' ', '')
        else:
            captions = captions.replace(self.init_prompt.lower() + ' ', '')
        return captions