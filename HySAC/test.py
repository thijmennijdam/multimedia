from hysac.models import HySAC
from transformers import CLIPTokenizer

print("Loading model...")

model_id = "aimagelab/hysac"

model = HySAC.from_pretrained(model_id, device="cpu")
print("Model loaded succesfully!")

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

# TODO: Use with desired textual data
input_text = "Hello, world!"

tokenized_text = tokenizer(input_text, return_tensors="pt", padding='max_length', truncation=True)
print("Tokenized text:", tokenized_text['input_ids'])

embed_text = model.encode_text(tokenized_text['input_ids'], project=True)

print("Embedded text shape:", embed_text.shape)

