import torch
from transformers import BertModel, BertConfig

# Load pretrained base BERT
pretrained_model = BertModel.from_pretrained('bert-base-uncased')

# Load the checkpoint you saved
finetuned_checkpoint = torch.load('./output/kaggle/working/assets/best_bert_model.pt', map_location='cpu')

# Now create a BERT model with same architecture
finetuned_model = BertModel(BertConfig.from_pretrained('bert-base-uncased'))

# Load your fine-tuned weights into it
finetuned_model.load_state_dict(finetuned_checkpoint)

# Store deltas
delta = {}

# Compare parameters
for (name_pretrained, param_pretrained), (name_finetuned, param_finetuned) in zip(
    pretrained_model.named_parameters(), finetuned_model.named_parameters()
):
    if name_pretrained != name_finetuned:
        raise ValueError(f"Layer name mismatch: {name_pretrained} != {name_finetuned}")
    
    # Compute delta
    delta[name_pretrained] = param_finetuned.data - param_pretrained.data

# Example: print delta for the first few layers
for name, d in list(delta.items())[:5]:
    print(f"Delta for {name}: mean={d.abs().mean().item():.6f}, max={d.abs().max().item():.6f}")

# Optional: Save deltas
torch.save(delta, 'bert_weight_deltas.pt')
