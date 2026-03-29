# Advisor Commission Bias Dashboard — Deployment Guide

## Project Structure

```
app/
├── streamlit_app.py                        ← Main Streamlit app
├── bias_model.py                           ← Model logic (loads from Excel)
├── requirements.txt                        ← Python dependencies
├── advisor_commission_bias_datasource.xlsx ← Data file (place here)
└── README.md
```

---

## Option 1 — Run Locally

```bash
# 1. Clone / copy the app/ folder
cd app

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the datasource file in the same folder as streamlit_app.py
cp /path/to/advisor_commission_bias_datasource.xlsx .

# 5. Launch
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**

---

## Option 2 — Deploy to Streamlit Community Cloud (free)

1. Push the `app/` folder to a **public GitHub repo**
2. Add the `.xlsx` file to the repo (or use the file uploader in the sidebar)
3. Go to **https://share.streamlit.io** → "New app"
4. Select your repo, branch `main`, and set the main file to `streamlit_app.py`
5. Click **Deploy** — your app gets a public URL in ~2 minutes

> **Tip:** Add a `[secrets]` section in Streamlit Cloud settings if you later
> add authentication or API keys.

---

## Option 3 — Deploy to AWS EC2 / Any Linux Server

```bash
# On the server
sudo apt update && sudo apt install python3-pip python3-venv -y

git clone https://github.com/YOUR_ORG/commission-bias-app.git
cd commission-bias-app/app

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on port 8080, accessible externally
streamlit run streamlit_app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --server.headless true

# To keep running after logout
nohup streamlit run streamlit_app.py \
  --server.port 8080 --server.address 0.0.0.0 --server.headless true \
  > streamlit.log 2>&1 &
```

Open Security Group / firewall port **8080**.

---

## Option 4 — Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

```bash
docker build -t commission-bias-app .
docker run -p 8501:8501 commission-bias-app
```

---

## Option 5 — Azure App Service / Google Cloud Run

For managed cloud deployment:

```bash
# Build and push Docker image
docker build -t commission-bias-app .
docker tag commission-bias-app gcr.io/YOUR_PROJECT/commission-bias-app
docker push gcr.io/YOUR_PROJECT/commission-bias-app

# Deploy to Cloud Run
gcloud run deploy commission-bias-app \
  --image gcr.io/YOUR_PROJECT/commission-bias-app \
  --platform managed \
  --port 8501 \
  --allow-unauthenticated
```

---

## Uploading a New Datasource

The sidebar has a **file uploader** — users can upload a fresh
`advisor_commission_bias_datasource.xlsx` at runtime without redeployment.
The model automatically retrains on the new data.

---

## Environment Variables (optional)

| Variable          | Default                                       | Purpose                        |
|-------------------|-----------------------------------------------|--------------------------------|
| `DATA_PATH`       | `advisor_commission_bias_datasource.xlsx`     | Override default data file     |
| `STREAMLIT_PORT`  | `8501`                                        | Port override                  |
