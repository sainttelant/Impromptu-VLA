If you want to create our benchmark QA data from scratch:

1. First, organize the data download based on `data_raw`.
2. Parse the data according to the code and instructions in the folder (for the `waymo` and `mapillary_sls` datasets).
3. Enter the main directory.Create a symbolic link for `navsim`:
   ```bash
   ln -s /data_raw/navsim /data_qa_generate/data_engine/data_storage/external_datasets/navsim
   ```
4. After the data is successfully organized, run the following script:
   ```bash
   bash scripts/data_qa_generate.sh
   ```
