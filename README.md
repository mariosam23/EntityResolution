# Entity Resolution for Company Data

This project provides a solution for the entity resolution problem, aimed at identifying duplicate company records in a dataset. It follows a structured pipeline involving data analysis, preprocessing, blocking, comparison, and clustering to produce a clean, deduplicated list of unique companies.

## Project Structure

- `entity_resolution.ipynb`: A Jupyter Notebook containing the step-by-step implementation and explanation of the entity resolution process.
- `src/`: Contains the Python source code.
  - `CompaniesDataAnalyzer.py`: A class for loading, analyzing, and preprocessing the company data.
  - `CompaniesEntityResolution.py`: A class that implements the core entity resolution logic, including blocking, comparison, and clustering.
- `input/`: Directory for the input data file.
- `output/`: Directory for the generated output files.
- `requirements.txt`: A list of python dependencies required to run the project.

## Methodology

The entity resolution process is implemented in the following stages:

### 1. Data Analysis and Feature Selection
- The dataset is loaded and analyzed to understand its structure and identify issues like missing values.
- Key features for identifying duplicates are selected, including `company_name`, `website_domain`, `primary_phone`, address components, and `main_industry`.

### 2. Data Preprocessing
- **Handling Missing Values**: Missing values in key fields are filled using information from other relevant columns (e.g., populating `website_domain` from `website_url`).
- **Data Cleaning**: String fields like `company_name` and `main_street` are cleaned by:
  - Removing non-ASCII characters.
  - Normalizing to a consistent case.
  - Removing common legal suffixes (e.g., 'LLC', 'Inc.').

### 3. Blocking
- To optimize the matching process and avoid comparing every record with every other record (O(n²)), a blocking strategy is used.
- Records are grouped into blocks based on:
  - `website_domain`
  - `company_name`
  - `primary_phone`
- This reduces the number of comparisons to only pairs within the same block.

### 4. Pairwise Comparison
- Within each block, candidate pairs of records are compared.
- String similarity metrics are used to score the similarity of their attributes:
  - **Longest Common Subsequence (LCS)** for `company_name`.
  - **Jaro-Winkler** for `website_domain`, address fields, and `main_industry`.
  - **Exact Match** for `primary_phone`.

### 5. Classification
- A set of rules is applied to the similarity scores to classify whether a pair of records is a match. A match is determined if:
  1. `website_domain` and `company_name` are highly similar.
  2. `primary_phone` is identical and `company_name` is highly similar.
  3. `company_name` is identical, and both location and industry are highly similar.

### 6. Clustering
- A graph-based clustering algorithm is used to group all matching records.
- This approach ensures transitivity (if A matches B, and B matches C, then A, B, and C belong to the same cluster).
- Each resulting cluster represents a single, unique company.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Place the data file:**
    - Put the `companies_dataset.snappy.parquet` file into the `input/` directory.

3.  **Run the Jupyter Notebook:**
    - Open and run the `entity_resolution.ipynb` notebook to execute the entire pipeline.

## Output

The process generates two output files in the `output/` directory:

- `unique_companies.snappy.parquet`: A Parquet file containing the deduplicated dataset, with one representative record for each unique company.
- `company_clusters.json`: A JSON file that maps each original record's index to its corresponding cluster ID. This allows for tracing records back to their identified unique entity.
