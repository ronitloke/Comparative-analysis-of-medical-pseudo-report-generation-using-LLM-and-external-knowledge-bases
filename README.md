# 🏥 Comparative Analysis of Medical Pseudo-Report Generation Using LLM and External Knowledge Bases

## 📌 Project Overview
Automated radiology report generation is a significant advancement in medical imaging, aimed at reducing radiologist's workload and enhancing diagnostic accuracy. This project presents an integrated approach that combines **chest X-ray images, free-text radiology reports, and external knowledge bases** to generate high-quality pseudo-reports using **Large Language Models (LLMs)** such as **GPT-4 and GPT-4O**.

### 🔬 Key Features:
- 📸 **Utilizes Chest X-ray (CXR) images** for generating detailed radiology reports.
- 🏥 **Integrates structured knowledge from Radiopaedia** via triplet extraction.
- 🖼️ **Employs ALBEF (Align Before Fuse) Model** for image-text feature alignment.
- 📖 **Uses pre-trained models** like **ResNet-50** (for image features) and **BERT** (for text embeddings).
- 📝 **Generates high-quality pseudo-reports** using **GPT-4 and GPT-4O**.
- 📊 **Evaluation Metrics:** BERTScore, TF-IDF similarity for precision, recall, and F1-score analysis.

---

## 📂 Directory Structure:

📁 Comparative analysis of medical pseudo-report generation using LLM and external knowledge bases/  
│  
├── 📂 data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dataset & External Knowledge  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📂 external_knowledge/        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📚 External knowledge base     
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 Radiopedia.xlsx    	  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🏥 Radiopedia file for structured triplet extraction  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📂 sample_chest_xrays_Images/        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🖼️ Sample chest X-ray images for testing  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📂 sample_free_text_radiology_reports/  # 📜 Sample free-text radiology reports  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 dataset_link.txt           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🔗 Dataset link file for reference  
│  
├── 📂 documentation/                   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Project Documentation      
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 report.pdf                   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📖 Detailed project report  
│  
├── 📂 notebooks/                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Jupyter Notebooks for experiments  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 Server_Part_1.ipynb         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🧪 Data processing & feature extraction  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 Part_2.ipynb                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📝 Report generation & evaluation  
│  
├── 📂 output/                           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Outputs generated from the model  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 cleaned_generated_reports.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#🧹 Cleaned version of generated reports  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 comparison_scores.csv         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📊 Comparison scores for AI-generated reports using BERTScore and TF-IDF  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 evaluation_results.csv        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📊 Evaluation results comparing AI-generated and ground truth reports  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 filtered_reports_with_ai.csv  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🔍 Filtered dataset with AI-generated reports merged  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📄 generated_reports.txt         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📝 Generated reports from AI model  
│  
├── 📂 scripts/                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Codebase for processing, embedding, & training  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 clean_reports.py            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🧹 Cleaning and preprocessing script for generated reports  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 dataset.py                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📊 Dataset processing script for loading and formatting data  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 embeddings.py               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🧠 Embeddings generation script for text and image data  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 evaluation.py               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📈 Evaluation script for assessing model performance  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 gcs_utils.py                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ☁️ Google Cloud Storage utilities for file handling  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 generate_report.py          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📝 Report generation script using AI model  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 inference.py                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🔍 Inference script for making predictions on new X-ray images  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 model.py                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🖥️ Model definition for multimodal report generation  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 train.py                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🎯 Model training script for supervised learning  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 triplets.py                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🔗 Triplet extraction script for knowledge integration  
│   &nbsp;&nbsp;&nbsp;&nbsp;├── 📝 utils.py                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 🛠️ Utility functions for common operations across the project  
│  
├── 📄 README.md                         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📖 Project Overview  
├── 📄 requirements.txt                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 📦 Dependencies for running the project  

## 🚀 How to Run the Project
### 1️⃣ Clone the repository
```bash
git clone https://github.com/ronitloke/Comparative-analysis-of-medical-pseudo-report-generation-using-LLM-and-external-knowledge-bases.git
cd Comparative-analysis-of-medical-pseudo-report-generation-using-LLM-and-external-knowledge-bases
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Download and Prepare Data
```bash
📄 data/dataset_link.txt
📂 data/sample_chest_xrays_Images/
📂 data/sample_free_text_radiology_reports/
📂 data/external_knowledge/Radiopedia.xlsx
```
### 4️⃣ Run Jupyter Notebook
```bash
# For Data Processing & Feature Extraction
📄 notebooks/Server_Part_1.ipynb
# For Report Generation & Evaluation
📄 notebooks/Part_2.ipynb
```

---

## 🏗 Methodology
### 🔹 Data Preparation
- **Dataset:** MIMIC-CXR dataset (chest X-ray images + corresponding free-text radiology reports).
- **Preprocessing:**  
  - **DICOM Processing:** Convert medical images to a normalized format using `pydicom`.
  - **Text Cleaning:** Remove noise, unnecessary whitespace, and irrelevant special characters.

### 🔹 Feature Extraction
- **Image Features:** Extracted via **ResNet-50** and converted into embeddings.
- **Text Features:** Extracted via **BERT**, tokenized using `transformers` library.
- **Alignment with ALBEF:** Contrastive learning aligns **image and text features** in a shared space.

### 🔹 Knowledge Integration using Radiopaedia
- **Triplet Extraction:** Structured knowledge extraction from **Radiopaedia articles**.
- **Embedding Generation:** Convert triplets into text embeddings via **BERT**.
- **Similarity Retrieval:** Compute **cosine similarity** to retrieve **top-5 relevant embeddings**.

### 🔹 Radiology Report Generation
- **GPT-4 & GPT-4O** are used to generate pseudo-reports using structured prompts:
  - **Direct Prompt:** Uses extracted text & image embeddings to generate a concise report.
  - **Few-Shot Prompting:** Uses examples from real reports to improve coherence & relevance.

### 🔹 Evaluation Metrics
- **BERTScore:** Measures semantic similarity (Precision, Recall, F1-score).
- **TF-IDF Cosine Similarity:** Measures term frequency and lexical similarity.

---

## 📊 Results
A comparative study between **GPT-4 and GPT-4O** was conducted:

| **Metrics**       | **GPT-4 (Direct Prompt)** | **GPT-4O (Direct Prompt)** | **GPT-4 (Few-Shot Prompt)** | **GPT-4O (Few-Shot Prompt)** |
|------------------|------------------------|------------------------|------------------------|------------------------|
| **BERTScore Precision** | 0.831345 | 0.841910 | 0.836496 | 0.857200 |
| **BERTScore Recall**    | 0.827029 | 0.821181 | 0.818821 | 0.818081 |
| **BERTScore F1**       | 0.829072 | 0.831197 | 0.827462 | 0.837124 |
| **TF-IDF Score**       | 0.276860 | 0.218646 | 0.190207 | 0.183484 |

### 🔹 Findings:
- **GPT-4 generated more detailed reports**, closely aligned with the original text.
- **GPT-4O produced concise and to-the-point summaries**, making it useful for quick interpretations.
- **Few-shot prompting improved report coherence & medical terminology precision**.

---

## 🔮 Future Work
&nbsp;   1. **Validation with Radiologists:** Ensure AI-generated reports meet clinical standards.  
&nbsp;   2. **Expansion to Other Modalities:** Apply the framework to CT scans, MRI, and Ultrasounds.  
&nbsp;   3. **Improved Model Interpretability:** Enhance explainability for clinical adoption.  
&nbsp;   4. **Integration into Real-Time Workflows:** Implement in hospitals for live AI-assisted report generation.  
&nbsp;   5. **Personalized Reports:** Use patient history & prior reports to tailor outputs.  

---

## 📜 Acknowledgments 

This project was conducted at **Dublin City University**, leveraging **MIMIC-CXR**, **Radiopaedia**, and state-of-the-art **LLMs**.

---

## 💡 Contribution Guidelines
```bash
git clone https://github.com/ronitloke/Comparative-analysis-of-medical-pseudo-report-generation-using-LLM-and-external-knowledge-bases.git
git checkout -b feature-branch
git commit -m "Your changes"
git push origin feature-branch
```

---

## 🚀 Follow for Updates
🌟 If you like this project, please star this repository!  
📬 Contact: ronitloke10@gmail.com

---






