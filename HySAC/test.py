from hysac.models import HySAC
from transformers import CLIPProcessor, CLIPTokenizer
from torch.cuda import is_available
from PIL import Image


class test_HySAC:
    def __init__(self):
        self.device = "cuda" if is_available() else "cpu"
        self.model_id = "aimagelab/hysac"
        self.clip_backbone = "openai/clip-vit-large-patch14"
        
        print("Loading model...")
        self.model = HySAC.from_pretrained(self.model_id, device=self.device).to(self.device)
        print("Model loaded succesfully!")
        
    def test_text_embedding(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.clip_backbone)
        
        # TODO: Use with desired textual data
        input_text = "Hello, world!"

        tokenized_text = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True).to(self.device)
        
        print("Tokenized text:", tokenized_text['input_ids'])
        embed_text = self.model.encode_text(tokenized_text['input_ids'], project=True)
        print("Embedded text shape:", embed_text.shape)
    
    def test_image_embedding(self):
        image_processor = CLIPProcessor.from_pretrained(self.clip_backbone)
        
        img_path = "../imgs/beach.jpg"
        
        image = Image.open(img_path)
        image = image_processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        
        print("Image tensor shape:", image.shape)
        embed_image = self.model.encode_image(image, project=True)
        print("Embedded image shape:", embed_image.shape)
        


if __name__ == "__main__":
    tester = test_HySAC()
    tester.test_text_embedding()
    tester.test_image_embedding()