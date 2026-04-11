param(
  [string]$ModelName = "meta-llama/Llama-2-7b-hf",
  [int]$Epochs = 3,
  [int]$Rank = 16,
  [int]$Alpha = 32,
  [double]$Dropout = 0.05
)

python .\train.py `
  --model_name $ModelName `
  --dataset_path .\dataset.json `
  --output_dir .\StudentID_Name_model `
  --report_dir .\outputs `
  --split_dir .\data_splits `
  --epochs $Epochs `
  --rank $Rank `
  --alpha $Alpha `
  --dropout $Dropout
