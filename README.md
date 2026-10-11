# Online Shopping Behaviour Analysis and Purchase Intention

A Python machine-learning project exploring online shopping session behaviour through **supervised classification** and **unsupervised clustering**. The notebooks cover data exploration and preprocessing, purchase-intention prediction, model evaluation, dimensionality reduction, and customer-session segmentation.

## Project Objectives

- Explore the online shopping session dataset and understand its features.
- Practice data cleaning and preprocessing, including missing-value handling, duplicate checks, outlier treatment, encoding, scaling, and feature engineering.
- Compare supervised machine-learning models for predicting purchase intention.
- Evaluate classification performance using metrics and visualizations.
- Explore behavioural groups using several clustering approaches.
- Visualize and interpret clusters, including how cluster assignments relate to purchase behaviour.

## Notebooks

### 1. Supervised Classification
`notebooks/01_supervised_classification.ipynb`

This notebook explores the dataset, prepares features, and compares the following classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Evaluation and analysis include classification reports, confusion matrices, ROC curves/AUC, model comparison visualizations, feature importance, permutation importance, and partial-dependence analysis where included in the notebook.

### 2. Unsupervised Customer Clustering
`notebooks/02_unsupervised_customer_clustering.ipynb`

This notebook explores customer-session patterns without using the purchase target as a clustering input. Methods and analysis include:

- K-Means
- Agglomerative/Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)
- Principal Component Analysis (PCA)
- t-SNE visualization
- Elbow and silhouette analysis
- Cluster profiling and comparisons with purchase behaviour

The notebook also includes clustering comparison metrics such as Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), where applicable in the existing workflow.

## Dataset

The project uses the **Online Shoppers Purchasing Intention** dataset.

- Dataset information: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- The notebooks expect the dataset file to be available in the working environment. Check the CSV filename and file path in the data-loading cell before running.
- The dataset is not included in this repository package. Download it from the source above and follow the source's terms before sharing or redistributing it.

## Tools and Libraries

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn
- SciPy
- Jupyter Notebook

Install the dependencies listed in `requirements.txt`.

## Getting Started

1. Clone or download this repository.
2. Install Python (Python 3.10 or newer is recommended).
3. Create and activate a virtual environment if desired.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Download the dataset from the source above and place it where the notebook's data-loading cell expects it.
6. Open the notebooks in Jupyter or VS Code and run the cells in order.

## Repository Structure

```text
online-shopping-behaviour-analysis/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── 01_supervised_classification.ipynb
    └── 02_unsupervised_customer_clustering.ipynb
```

## Notes

- The notebooks preserve the existing project workflow; the filenames and repository documentation have been organized for portfolio presentation.
- Some preprocessing cells may introduce artificial missing values for demonstration. These should be understood as simulated missingness rather than necessarily missing values in the original dataset.
- Results can depend on library versions, random seeds, and execution order. Run the notebooks from top to bottom to reproduce their outputs.
- No performance score is stated here; consult the executed notebooks for the results produced by their current code.

## Author

**Sara Rasoli**  
Computer Science | Data Analytics | Machine Learning

GitHub: https://github.com/Sara-Rasoli
