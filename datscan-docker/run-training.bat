@echo off
echo Starting training in Docker container...
echo.

REM Set default configuration
set DATA_CSV=data/validated_file_paths.csv
set BATCH_SIZE=8
set GRADIENT_ACCUMULATION=4
set NUM_WORKERS=6
set CACHE_DIR=processed_cache
set OUTPUT_DIR=trained_models/autoencoder

echo Using the following configuration:
echo - Data CSV: %DATA_CSV%
echo - Batch size: %BATCH_SIZE%
echo - Gradient accumulation: %GRADIENT_ACCUMULATION%
echo - Number of workers: %NUM_WORKERS%
echo - Cache directory: %CACHE_DIR%
echo - Output directory: %OUTPUT_DIR%
echo.

echo Creating output directory structure...
mkdir %OUTPUT_DIR%\checkpoints 2>nul
mkdir %CACHE_DIR% 2>nul
echo.

echo Running Docker container...
docker-compose exec datscan python -c "import os; print('Current directory:', os.getcwd()); print('Directory contents:', os.listdir('.')); print('Data directory contents:', os.listdir('data'))"

echo Checking file existence...
docker-compose exec datscan python -c "import pandas as pd; import os; df = pd.read_csv('%DATA_CSV%'); print(f'First 3 file paths:'); for i, row in df.head(3).iterrows(): print(f\"  {row['file_path']} - Exists: {os.path.exists(row['file_path'])}\")"

echo.
echo Starting training...
docker-compose exec datscan python train.py --data_csv %DATA_CSV% --batch_size %BATCH_SIZE% --gradient_accumulation %GRADIENT_ACCUMULATION% --num_workers %NUM_WORKERS% --cache_dir %CACHE_DIR% --output_dir %OUTPUT_DIR%

echo.
echo Training complete!