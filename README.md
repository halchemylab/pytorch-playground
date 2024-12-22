# HR Matching System: README

## Overview

This project demonstrates a simplified candidate-job matching system using machine learning with PyTorch. While the dataset is artificially generated, the goal is to simulate a recruitment scenario where machine learning models predict compatibility between candidates and job openings.

The core idea is to:
- Build a dataset of candidates and job orders.
- Train a model to predict whether a candidate is a "fit" for a job.
- Provide an entry point for learning PyTorch and applying it to real-world problems like HR recruitment systems.

---

## How It Works

### 1. Dataset Generation

Two datasets are generated:

#### Candidates Dataset:
Contains candidate profiles with the following features:
- **Skills**: A list of skills the candidate possesses.
- **ExperienceYears**: Total years of professional experience.
- **Education**: Highest degree earned (e.g., Bachelor's, Master's, PhD).
- **CurrentLocation**: Candidate's current city.
- **RelocationWillingness**: Whether the candidate is open to relocating.
- **SalaryExpectation**: Candidate's expected salary range.
- **PersonalityAssessment**: A general personality type.

#### Job Orders Dataset:
Contains job openings with the following features:
- **RequiredSkills**: A list of skills required for the job.
- **JobTitle**: The job role title (e.g., Software Engineer).
- **JobDescription**: A textual description of the role and its requirements.
- **Location**: Location of the job.
- **SalaryRange**: Salary offered for the position.
- **CompanyInfo**: Industry or other relevant company details.
- **JobType**: Type of job (Full-time, Part-time, Contract).

---

### 2. Machine Learning Pipeline

#### Feature Preprocessing
- **Skills Representation**: Skills are one-hot encoded to create feature vectors.
- **Label Generation**: Labels (`fit` or `no fit`) are generated based on an overlap threshold between candidate and job-required skills.

#### Model Architecture
A simple neural network is implemented in PyTorch with the following structure:
- **Input Layer**: Encoded feature vectors.
- **Hidden Layers**: Dense layers with ReLU activation.
- **Output Layer**: A single neuron with Sigmoid activation for binary classification.

#### Training and Evaluation
- **Training**: Binary cross-entropy loss is used with the Adam optimizer.
- **Metrics**: Accuracy is calculated to evaluate performance. For imbalanced datasets, F1 score, precision, and recall are recommended.

---

## Limitations
- **Artificial Nature of Data**: The dataset is randomly generated and does not reflect real-world recruitment scenarios.
- **Simplistic Features**: Complex factors like personality traits, company culture, or past hiring trends are not included.
- **Label Randomness**: Initial labels are random, which limits the model's ability to learn meaningful patterns.

---

## Next Steps for Improvement

### Dataset:
- Use real-world data to capture meaningful relationships.
- Add additional features like resume text, company culture, and hiring trends.

### Feature Representation:
- Incorporate embeddings for skills and textual data (e.g., job descriptions).
- Use advanced NLP models like BERT for semantic understanding.

### Model Architecture:
- Experiment with deeper networks or transformer-based architectures.
- Implement recommendation systems for ranking candidates or jobs.

### Evaluation Metrics:
- Evaluate with F1 score, precision, and recall to handle imbalanced data better.

---

## Getting Started

### 1. Install Dependencies
```bash
pip install torch torchvision pandas scikit-learn
