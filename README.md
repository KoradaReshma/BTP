
# Network Traffic Analysis and Botnet Detection System

## Overview
This Jupyter notebook implements a comprehensive multi-layer system for network traffic analysis and botnet detection using machine learning and deep learning approaches. The system processes network flow data to classify benign traffic and detect various types of botnet attacks through a hierarchical classification approach.

## Key Components

### Data Processing and Feature Engineering

#### 1. Data Cleaning Function
clean_numeric_df(df, target_column='label')
- Removes duplicate entries
- Handles infinite values in numeric columns
- Strategically fills missing values (median for skewed data, mean for normal, mode for categorical)
- Applies IQR capping for outlier detection and treatment
- Returns cleaned DataFrame

#### 2. Feature Set Definitions
- **Set A Features**: Flow time & rate features, inter-arrival times, packet size statistics, active/idle patterns, subflow dynamics
- **Set B Features**: Protocol and ports, flag-based patterns, flow-level packet stats, header and ratio cues
- **Set A+B Features**: Combined feature set from both A and B

#### 3. Feature Management Functions
```python
ensure_features(df: pd.DataFrame, req_set) -> pd.DataFrame
```
- Ensures required feature sets are present in the DataFrame
- Adds missing features with default value of 0

```python
get_feature_set(df: pd.DataFrame, set_list) -> pd.DataFrame
```
- Extracts specified feature sets from the DataFrame

```python
features_union(set_list)
```
- Returns union of features from specified sets

```python
process_file(file_path: str, set_list) -> pd.DataFrame
```
- Complete file processing pipeline: read, ensure features, filter, and clean data

### Data Preparation

#### 4. Data Loading and Splitting
- Loads benign traffic data and splits into training/test sets
- Processes attack data from multiple CSV files
- Creates hierarchical datasets for different classification layers:
  - **Layer 1**: Benign vs Attack classification
  - **Layer 2**: Botnet family classification (neris, rbot, attack pattern neither neris or r-bot)
  - **Layer 3**: Specific botnet classification (menti, murlo, nsisay, virut)
  - **Layer 4**: Zero-day attack detection

### Machine Learning Models

#### 5. Autoencoder for Anomaly Detection (Layer 1)
```python
build_autoencoder(input_dim, latent_dim=10)
```
- Constructs encoder-decoder architecture for reconstruction-based anomaly detection
- Encoder: Dense layers (64 → 32 → latent_dim)
- Decoder: Dense layers (32 → 64 → input_dim)

```python
train_layer1_autoencoder(df, latent_dim=10)
```
- Trains autoencoder on benign traffic data
- Uses StandardScaler for feature normalization
- Implements early stopping and learning rate scheduling
- Returns trained model, decoder, scaler, training history, and reconstruction errors

#### 6. CNN-Bidirectional LSTM for Botnet Classification (Layer 2)
```python
train_layer2_model_sparse(df_attack_layer_2, epochs=50, batch_size=32, verbose=1)
```
- Hybrid architecture: Conv1D → Bidirectional LSTM → Dense classifier
- Processes features as temporal sequences
- Uses sparse categorical crossentropy loss
- Returns trained model, scaler, and label encoder

#### 7. Random Forest for Specific Botnet Classification (Layer 3)
```python
train_layer3_rf(df_attack)
```
- Traditional Random Forest classifier
- Handles multi-class classification for specific botnet types
- Returns trained model and label encoder

#### 8. VAE for Zero-Day Detection (Layer 4)
```python
build_four_vaes(df_attack)
```
- Trains separate VAE models for each botnet class
- Enables reconstruction error-based anomaly detection for unknown attacks

### Prediction and Evaluation

#### 9. Detection Functions
```python
detect_attacks_layer1(test, vae_model, scaler, threshold=0.1)
```
- Uses MAE reconstruction error for attack detection
- Computes confusion matrix and performance metrics
- Returns detected attack samples

```python
predict_layer2_model(model, scaler, label_encoder, data, feats_list)
```
- Performs botnet family classification
- Provides detailed performance analysis including detection rates and false alarm rates
- Routes "other" class samples to Layer 3

```python
predict_layer3_simple(rf_model, label_encoder, layer2_output_df, feats_list)
```
- Classifies specific botnet types
- Generates confusion matrix and classification report

```python
layer4_zero_day_detection(layer3_results, vae_models, scalers, threshold=0.2)
```
- Detects zero-day attacks using reconstruction error thresholding

### Utility Functions

#### 10. SHAP Analysis
```python
shap_dashboard(rf_model, x_data)
```
- Provides model interpretability using SHAP values
- Generates feature importance plots and analysis

#### 11. Comprehensive Model Training
```python
train_layer2_new_all_features(df_attack, feat_list)
```
- Enhanced Random Forest training with all available features on all attacks
- Handles complete botnet classification task

## Workflow
1. **Data Preprocessing**: Clean and normalize network traffic data
2. **Feature Extraction**: Select relevant feature sets for different detection tasks
3. **Hierarchical Classification**:
   - Layer 1: Autoencoder detects anomalies (benign vs attack)
   - Layer 2: CNN-LSTM classifies botnet families
   - Layer 3: Random Forest identifies specific botnet types
   - Layer 4: VAE detects zero-day attacks
4. **Performance Evaluation**: Comprehensive metrics and confusion matrices
5. **Model Interpretation**: SHAP analysis for feature importance

## Supported Botnet Types
- **Neris**: IRC-based botnet
- **Rbot**: Remote access trojan
- **Menti**: Specific malware variant
- **Murlo**: Backdoor trojan
- **Nsisay**: Information stealer
- **Virut**: File infector and spam bot

## Requirements
- TensorFlow/Keras
- Scikit-learn
- Pandas/Numpy
- SHAP (for model interpretation)
- Matplotlib/Seaborn (for visualization)

This system provides a robust framework for network security analysis, capable of detecting both known and unknown botnet activities through its hierarchical multi-model approach.
```
