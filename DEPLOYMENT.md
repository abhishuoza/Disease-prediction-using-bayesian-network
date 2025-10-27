# Deployment Instructions for Render

## Quick Start

### 1. Train the Model (First Time Only)
Before deploying, train the Bayesian Network model:
```bash
python train_model.py
```
This creates `trained_model.pkl` (40KB) which will be committed to the repository.

### 2. Deploy to Render

#### Create a Render Account
1. Go to https://render.com
2. Sign up with your GitHub account (free)

#### Deploy Using render.yaml (Recommended)
1. Log into Render dashboard
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will detect `render.yaml` automatically
5. Click **"Apply"**
6. Wait 5-10 minutes for deployment

Your app will be live at: `https://disease-predictor.onrender.com`

## Important Notes

### Model Architecture
- Pre-trained model (40KB) loads in ~2-4 seconds at startup
- Training is separated from deployment for faster cold starts
- Model predicts 13 diseases from 57 symptoms using Bayesian Network

### Free Tier Behavior
- Apps spin down after 15 minutes of inactivity
- First visit takes ~30-60 seconds to wake up (normal)
- 750 hours/month included (enough for portfolio projects)

### Monitoring
View logs in Render dashboard to see:
- Model loading progress
- Prediction requests
- Any errors

### Auto-Deploy
Render automatically redeploys when you push to GitHub (can be disabled in Settings)

## Troubleshooting

**Build fails:** Ensure `trained_model.pkl` exists in repository
**App won't start:** Check logs for missing dependencies
**Predictions fail:** Verify `Diseases.csv` exists in `Website/` folder

---

**Your ML app is now deployed and accessible worldwide!**
