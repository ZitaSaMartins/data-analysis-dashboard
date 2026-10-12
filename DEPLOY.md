# Deployment Guide: Streamlit Cloud (Free!)

This guide walks you through deploying the Data Analysis Dashboard to **Streamlit Cloud** with zero cost.

## Prerequisites

- GitHub account (free: https://github.com)
- Your code repo pushed to GitHub

## Step-by-Step Deployment

### Step 1: Prepare Your GitHub Repository

1. Initialize Git (if not already done):
```bash
git init
git config user.name "Your Name"
git config user.email "your.email@gmail.com"
```

2. Add all files:
```bash
git add .
git commit -m "Initial commit: Data Analysis Dashboard"
```

3. Add GitHub remote:
```bash
git remote add origin https://github.com/YOUR-USERNAME/data-analysis-dashboard.git
```

4. Push to GitHub:
```bash
git branch -M main
git push -u origin main
```

**Verify:** Go to https://github.com/YOUR-USERNAME/data-analysis-dashboard and confirm files are there.

---

### Step 2: Deploy to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**

2. Click the **"New app"** button (top-left)

3. **Sign up with GitHub**:
   - Click "Sign in with GitHub"
   - Authorize Streamlit to access your GitHub repos
   - (Do this only once)

4. **Deploy your app**:
   - **Repository**: Select `YOUR-USERNAME/data-analysis-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - Click **"Deploy"**

5. **Wait for deployment** (1-2 minutes)
   - Streamlit will build and deploy your app
   - You'll see a progress bar at the top

6. **Your app is live!** 🎉
   - URL format: `https://YOUR-USERNAME-data-analysis-dashboard.streamlit.app`
   - Share this URL with your students

---

## Sharing with Students

### Option 1: Direct Link (Easiest)
Share the Streamlit Cloud URL:
```
https://YOUR-USERNAME-data-analysis-dashboard.streamlit.app
```

Students can:
- Click the link
- Upload their data
- Run analysis
- Download results

No installation needed!

### Option 2: Through Your Course Portal
Add the link to:
- Canvas / Blackboard / Moodle
- Your course website
- Email syllabus

### Option 3: QR Code
Generate a QR code for the URL and display it in class.

---

## Updating Your App

When you make changes:

1. **Make changes locally**:
```bash
# Edit files (e.g., app.py, modules/*, etc.)
```

2. **Commit and push**:
```bash
git add .
git commit -m "Description of changes"
git push
```

3. **Streamlit Cloud auto-updates**:
   - Automatically redeploys within 1-2 minutes
   - No manual action needed!

---

## Troubleshooting

### App won't load

**Error: "Failed to connect"**
- Wait 2-3 minutes for deployment
- Refresh the page
- Check app logs (Streamlit Cloud dashboard)

**Error: "Import Error"**
- Verify all dependencies are in `requirements.txt`
- Check for typos in file names
- Ensure Python files are in the right folders

### Slow performance

- Streamlit Cloud free tier can be slower than local
- File uploads may take longer
- First load after inactivity takes ~1 min

**Solutions:**
- Keep datasets under 50MB
- Optimize code (avoid loops, cache expensive operations)
- Upgrade to Streamlit Cloud paid tier if needed

### How to check logs

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. See "Logs" tab for error messages

---

## Optional: Streamlit Cloud Settings

### 1. Private Deployment (Paid Feature)
- Default: App is public
- To make private: Upgrade to Streamlit Community Cloud (paid)

### 2. Custom Domain (Paid Feature)
- Use your own domain instead of streamlit.app
- Requires paid tier

### 3. App Secrets (Advanced)
For sensitive data (API keys, etc.):

1. Go to Streamlit Cloud app settings
2. Add secrets (environment variables)
3. Access in code: `st.secrets["secret_name"]`

---

## Limits & Quotas

| Feature | Free Tier | Notes |
|---------|-----------|-------|
| Number of apps | 3 | Upgrade to deploy more |
| Concurrent users | ∞ | Unlimited |
| Inactivity timeout | 1 hour | App sleeps, wakes on access |
| Storage | None | Upload/download only |
| Compute | Shared | Slower than dedicated |
| Cold start | ~1-2 min | After sleep |

---

## Example: Full Deployment Workflow

```bash
# 1. Create repo (first time only)
git init
git remote add origin https://github.com/myusername/data-analysis-dashboard.git

# 2. First push
git add .
git commit -m "Initial: Data Analysis Dashboard"
git push -u origin main

# 3. Deploy to Streamlit Cloud
# → Go to share.streamlit.io and deploy

# 4. Later: Make updates
# → Edit files locally
git add .
git commit -m "Added new statistical test"
git push
# → Streamlit Cloud auto-redeploys!
```

---

## Advanced: Deploy Elsewhere (Optional)

If you want to avoid Streamlit Cloud limits:

### Heroku (Free)
- https://www.heroku.com (limited free tier after 2022)
- Need account + credit card (won't charge)

### AWS / Google Cloud
- More complex setup
- Better for high-traffic apps
- Not free, but cheap for education

### Your Own Server
- Docker deployment
- More control
- Requires Linux knowledge

**Recommendation:** Start with Streamlit Cloud (easiest), upgrade later if needed.

---

## Getting Help

**Streamlit Documentation:** https://docs.streamlit.io
**Streamlit Forum:** https://discuss.streamlit.io
**GitHub Issues:** https://github.com/YOUR-USERNAME/data-analysis-dashboard/issues

---

**You're ready to go! Deploy and share with your students! 🚀**