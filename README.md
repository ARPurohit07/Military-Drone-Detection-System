# Military Drone Detection System

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch%2FTensorFlow-orange)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-yellow)
![Status](https://img.shields.io/badge/status-research-lightgrey)

This repository contains code and experiments for detecting and classifying military drones using deep learning techniques. The system leverages computer vision models to identify drones across different categories based on collected imagery.

## Repository Structure

- **single_camera.ipynb**  
  Implementation and testing of drone detection using a single-camera setup.  

- **multi_camera.ipynb**  
  Experimental notebook exploring detection in multi-camera setups to improve accuracy and reduce blind spots.  

- **test3.ipynb**  
  Additional testing and evaluation notebook for refining the detection pipeline.  

## Dataset

The dataset used for this project was curated through custom web scraping and manual annotation.  

> **Note:** The dataset contains sensitive information about military drones and cannot be shared publicly.  
> If you wish to replicate the results, you will need to build your own dataset following ethical and legal guidelines.

## Features

- Drone detection and classification using deep learning  
- Multi-camera experimentation to enhance robustness  
- Transfer learning applied on pretrained CNN/RCNN architectures  
- Evaluation metrics for model performance  

## Requirements

- Python 3.9+
- Jupyter Notebook
- PyTorch / TensorFlow
- OpenCV
- NumPy, Pandas, Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
