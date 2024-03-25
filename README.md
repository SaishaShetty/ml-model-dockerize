# ML Model Dockerization

This repository demonstrates how to containerize a machine learning (ML) model using Docker. It includes scripts to train a simple linear regression model and perform inference on new data using the trained model.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone <repo_url> 
   ```
2. **Build Docker Image:**
```bash
   docker build -t <image-name> .
   ```

3. **Run Docker Container for Training:**
```bash
   docker run <image-name>
   ```
   This command executes the training script (train.py) inside the Docker container, training the linear regression model and saving it as linear_regression_model.joblib.

4. **Run Docker Container for Inference:**
```bash
   docker run <image-name> python inference.py
   ```
   This command loads the trained model and performs inference on a sample input, saving the predictions to output.csv.

**Project Structure:**

- `train.py`: Python script for training the linear regression model.
- `inference.py`: Python script for performing inference using the trained model.
- `Dockerfile`: Dockerfile for building the Docker image.
- `requirements.txt`: List of Python dependencies required for the project.

