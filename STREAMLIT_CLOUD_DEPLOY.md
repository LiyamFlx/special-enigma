# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

Your app is ready to deploy at: **https://slavinyahoo.streamlit.app/**

### Prerequisites
- GitHub repository with latest code (✅ Done!)
- Streamlit Cloud account connected to GitHub

### Deployment Steps

#### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

#### 2. Deploy New App
Click "New app" or "Deploy an app"

#### 3. Configure Deployment
- **Repository:** `LiyamFlx/special-enigma`
- **Branch:** `main`
- **Main file path:** `app.py`
- **App URL:** `slavinyahoo` (custom URL)

#### 4. Advanced Settings (Optional)
Click "Advanced settings" to configure:
- **Python version:** 3.11 (recommended)
- **Secrets:** None needed for basic functionality

#### 5. Deploy!
Click "Deploy" and wait for the build to complete (~5-10 minutes)

---

## Files Ready for Deployment

✅ **app.py** - Main Streamlit application
✅ **requirements.txt** - Python dependencies (cloud-optimized)
✅ **packages.txt** - System packages (FFmpeg for audio)
✅ **.streamlit/config.toml** - Streamlit configuration
✅ **video_generator.py** - Video processing module
✅ **audio_processor.py** - Audio processing module

---

## Expected Build Time
- **Initial deployment:** 5-10 minutes
- **Subsequent updates:** 2-5 minutes

---

## Resource Limits on Streamlit Cloud

### Free Tier Limits:
- **RAM:** 1 GB
- **CPU:** Shared
- **Storage:** Limited temporary storage
- **Execution time:** 10 minutes per request

### Optimizations Applied:
✅ Neural network disabled to save memory
✅ Optical flow used for all processing
✅ Frame batching for memory efficiency
✅ Video downscaling to 1080p max

---

## Monitoring Deployment

### Check Build Logs
1. Go to your app dashboard
2. Click "Manage app"
3. View logs in real-time

### Common Build Issues:

**Issue: Out of memory during build**
```
Solution: Dependencies are optimized. If still fails, try:
- Removing torch/torchvision temporarily
- Using CPU-only torch version
```

**Issue: FFmpeg not found**
```
Solution: packages.txt already includes ffmpeg
Verify it's being installed in build logs
```

**Issue: Import errors**
```
Solution: All modules are included in repo
Check requirements.txt has all dependencies
```

---

## Post-Deployment Testing

### Test Checklist:
1. ✅ App loads without errors
2. ✅ Upload interface appears
3. ✅ Video upload works (small test video first)
4. ✅ Extension processing completes
5. ✅ Download button works
6. ✅ Audio is synchronized (if enabled)

### Test Videos:
- Start with short videos (5-10 seconds)
- File size under 50MB recommended
- 720p or 1080p resolution

---

## Updating Your Deployed App

Streamlit Cloud auto-deploys when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update feature"
git push origin main

# App automatically redeploys!
```

---

## Custom Domain (Already Configured)

Your app will be accessible at:
- **https://slavinyahoo.streamlit.app/**

To change the subdomain:
1. Go to app settings in Streamlit Cloud
2. Click "General"
3. Edit "App URL"

---

## Secrets Management (If Needed Later)

For API keys or sensitive data:
1. Go to app settings
2. Click "Secrets"
3. Add in TOML format:
```toml
[api]
key = "your-secret-key"
```

Access in code:
```python
import streamlit as st
api_key = st.secrets["api"]["key"]
```

---

## Performance Tips for Cloud

### Video Processing:
- Limit input video to 30 seconds max
- Auto-downscale to 720p for cloud (1080p for local)
- Use "Fast" mode as default

### Audio Processing:
- Audio extension works well on cloud
- Phase vocoder is CPU-based (no GPU needed)

### Suggested Cloud Limits:
Add to app.py:
```python
# Cloud-specific limits
MAX_DURATION_CLOUD = 30  # seconds
MAX_EXTENSION_CLOUD = 5  # seconds
```

---

## Troubleshooting Cloud Deployment

### App Crashes During Processing
**Cause:** Memory limit exceeded
**Solution:** Processing already optimized for cloud

### Slow Processing
**Cause:** Shared CPU on free tier
**Solution:** Normal on cloud, ~2-3x slower than local

### Upload Fails
**Cause:** File too large
**Solution:** config.toml sets maxUploadSize=200MB

---

## Monitoring & Analytics

### Streamlit Cloud Dashboard:
- View app usage statistics
- Monitor errors and crashes
- Check resource usage

### Add Custom Analytics (Optional):
```python
# Add to app.py
import streamlit as st

# Track usage
if 'video_count' not in st.session_state:
    st.session_state.video_count = 0

st.session_state.video_count += 1
```

---

## Support & Resources

- **Streamlit Docs:** https://docs.streamlit.io/
- **Community Forum:** https://discuss.streamlit.io/
- **Cloud Docs:** https://docs.streamlit.io/streamlit-community-cloud

---

## Next Steps After Deployment

1. ✅ Share your app URL: https://slavinyahoo.streamlit.app/
2. ✅ Test with real videos
3. ✅ Monitor performance
4. ✅ Collect user feedback
5. ⚡ Upgrade to paid tier if needed (more resources)

---

**Your app is ready to deploy! Just push to GitHub and it will auto-deploy to Streamlit Cloud.**
