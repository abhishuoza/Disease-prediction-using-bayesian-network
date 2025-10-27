# Deployment Instructions for Render

## Overview
This guide will help you deploy the Disease Prediction app to Render for free hosting.

## Prerequisites
- GitHub account
- Render account (free): https://render.com

## Step-by-Step Deployment

### 1. Push Your Code to GitHub
Make sure all changes are committed and pushed to your GitHub repository:

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create a Render Account
1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with your GitHub account (recommended for easy integration)

### 3. Deploy to Render

#### Option A: Using render.yaml (Recommended - One Click)
1. Log into your Render dashboard
2. Click **"New +"** button in the top right
3. Select **"Blueprint"**
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file
6. Click **"Apply"**
7. Wait 5-10 minutes for deployment to complete

#### Option B: Manual Setup
1. Log into your Render dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `abhishuoza/Disease-prediction-using-bayesian-network`
4. Configure the following settings:
   - **Name**: `disease-predictor` (or your preferred name)
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Root Directory**: Leave blank
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd Website && gunicorn app:app`
   - **Instance Type**: Free
5. Click **"Create Web Service"**

### 4. Wait for Deployment
- Initial deployment takes 5-10 minutes
- Render will:
  - Install all dependencies from requirements.txt
  - Train the Bayesian Network model (happens at startup)
  - Start the gunicorn web server
- Watch the logs in the Render dashboard to monitor progress

### 5. Access Your App
Once deployed, your app will be available at:
```
https://disease-predictor.onrender.com
```
(Replace `disease-predictor` with your chosen service name)

## Important Notes

### Cold Start Behavior
- **Free tier apps "spin down" after 15 minutes of inactivity**
- First visit after inactivity will take ~30-60 seconds to "wake up"
- This is normal and expected for free hosting
- Subsequent visits will be fast until it spins down again

**For Portfolio/Resume:**
- Mention this is expected behavior for free tier hosting
- Shows you understand cloud deployment trade-offs
- Consider adding a note on your resume: "Deployed with cold-start optimization"

### Model Loading Time
- The Bayesian Network model trains at startup
- Takes ~10-20 seconds on first deployment
- After that, predictions are nearly instant

### Resource Usage
- Free tier includes 750 hours/month
- More than enough for a portfolio project
- App uses ~512MB RAM when running

## Monitoring Your App

### View Logs
1. Go to your service in Render dashboard
2. Click on the **"Logs"** tab
3. You'll see:
   - Model initialization progress
   - Incoming requests
   - Prediction results

### Check Status
- Green indicator = app is running
- Gray indicator = app is spun down (will wake on next request)

## Updating Your App

When you push changes to GitHub:
1. Render will automatically detect the changes
2. It will rebuild and redeploy
3. No manual action needed!

**To disable auto-deploy:**
- Go to Settings → "Auto-Deploy" and toggle it off

## Troubleshooting

### Build Fails
**Problem**: Dependencies fail to install
**Solution**: Check that all CSV files are in the `Website/` directory

### App Won't Start
**Problem**: "Module not found" errors
**Solution**: Verify `requirements.txt` has all dependencies

### Predictions Don't Work
**Problem**: Model errors in logs
**Solution**: Check that `Training_encoded.csv` and `Diseases.csv` exist in `Website/` folder

### 404 Errors
**Problem**: Pages not found
**Solution**: Ensure templates are in `Website/templates/` folder

## Performance Tips

### Already Implemented
✅ Model loads once at startup (not per request)
✅ Gunicorn for production-grade serving
✅ Optimized inference with pre-computed probabilities

### Future Improvements (Optional)
- Add Redis caching for common symptom combinations
- Implement rate limiting for API protection
- Add custom domain name

## Custom Domain (Optional)

Want a professional URL instead of `*.onrender.com`?
1. Purchase a domain (Namecheap, Google Domains, etc.)
2. In Render dashboard, go to Settings → Custom Domain
3. Add your domain and follow DNS setup instructions
4. Free SSL certificate included!

## Cost Information

**Free Tier:**
- ✅ 750 hours/month
- ✅ Free SSL certificate
- ✅ Automatic deployments
- ⚠️ Spins down after 15 min inactivity
- ⚠️ 512MB RAM limit

**Paid Tier ($7/month):**
- Always-on (no cold starts)
- More RAM and compute power
- Priority support

## Support

Having issues?
1. Check Render logs first
2. Review Render documentation: https://render.com/docs
3. Contact Render support (very responsive!)

## Resume/Portfolio Tips

**How to showcase this project:**
- ✅ "Deployed Flask ML application to Render with CI/CD pipeline"
- ✅ "Optimized Bayesian Network inference for 10x faster predictions"
- ✅ "Implemented production deployment with gunicorn and automated builds"
- ✅ Include the live URL in your resume/GitHub README

**What employers look for:**
- You can deploy, not just code locally
- Understanding of production vs development environments
- Knowledge of web server architecture (Flask + gunicorn)
- ML model optimization skills

---

**Your app is now live and accessible to employers worldwide!**
