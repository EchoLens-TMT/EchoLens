# import os
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm

# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# # Constants
# MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Custom Dataset
# class ImageCaptionDataset(Dataset):
#     def __init__(self, csv_file, image_dir, feature_extractor, tokenizer, max_length=64):
#         self.df = pd.read_csv(csv_file)
#         self.image_dir = image_dir
#         self.feature_extractor = feature_extractor
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.df.iloc[idx, 0])
#         caption = self.df.iloc[idx, 1]

#         image = Image.open(img_path).convert("RGB")
#         pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()

#         tokenized = self.tokenizer(
#             caption,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         ).input_ids.squeeze()

#         return {
#             "pixel_values": pixel_values,
#             "labels": tokenized
#         }

# # Load Model and Tokenizer
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
# feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model.to(DEVICE)

# # Dataset and DataLoader
# dataset = ImageCaptionDataset(
#     csv_file="/Users/tejfaster/Developer/Python/cv_project/Tej/EchoLens/DataSet/captions.csv",
#     image_dir="images",
#     feature_extractor=feature_extractor,
#     tokenizer=tokenizer
# )

# # Split dataset
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# # Training arguments
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./fine_tuned_model",
#     evaluation_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     save_total_limit=2,
#     fp16=torch.cuda.is_available(),
#     push_to_hub=False,
#     logging_dir='./logs',
# )

# # Data collator
# def collate_fn(batch):
#     pixel_values = torch.stack([x["pixel_values"] for x in batch])
#     labels = torch.stack([x["labels"] for x in batch])
#     return {"pixel_values": pixel_values, "labels": labels}

# # Trainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=collate_fn,
#     tokenizer=feature_extractor,
# )

# # Start training
# trainer.train()

# # Save model
# model.save_pretrained("fine_tuned_model")
# tokenizer.save_pretrained("fine_tuned_model")
