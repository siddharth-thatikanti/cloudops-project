# Cloud Infrastructure Cost Prediction — Linear Regression

Beginner ML/MLOps project: train a Scikit-learn Linear Regression model and serve predictions through FastAPI.

**No Dockerfile or Docker image is included.**

## Structure
- `dataset/cloud_cost.csv` — synthetic sample dataset
- `train.py` — model training and evaluation
- `app.py` — FastAPI prediction service
- `model/` — generated model file (created by training)
- `requirements.txt` — Python dependencies

## 1. Create a virtual environment (recommended)
Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows PowerShell:
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Train the model
Run from the project root:
```bash
python train.py
```
This prints MAE, RMSE and R-squared, then creates `model/cloud_cost_model.pkl`.

## 4. Start the API
```bash
uvicorn app:app --reload
```

Open:
- API: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

## 5. Test POST /predict
```json
{
  "cpu": 75,
  "memory": 65,
  "network": 250,
  "storage": 200,
  "running_hours": 650
}
```

## Important limitations
The CSV is synthetic demonstration data, not real AWS billing or CloudWatch data. The prediction is not a reliable AWS bill forecast. Real cloud cost depends on instance type, region, pricing model, data transfer, discounts, taxes, storage class and other factors. CPU and memory utilization do not directly determine EC2 charges. For a serious project, replace the sample data with historical billing and usage data and validate against a baseline.
