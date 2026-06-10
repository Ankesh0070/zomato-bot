# 🚀 Streamlit Deployment Plan

This document outlines the step-by-step instructions for deploying the **Zomato AI Restaurant Recommender** ("The Last Bite") on **Streamlit Community Cloud**.

---

## 📋 Prerequisites

Before proceeding, ensure you have:
1. A **GitHub Account** with the project repository uploaded.
2. A **Streamlit Community Cloud Account** (sign up at [share.streamlit.io](https://share.streamlit.io/)).
3. A **Groq API Key** (obtain a free key from the [Groq Console](https://console.groq.com/keys)).

---

## 🛠️ Step 1: Repository Preparation

To ensure Streamlit Community Cloud can build and run your application successfully, verify the following files in your repository structure:

### 1. The Entry Point
Streamlit needs to know which file to execute. The entry point for the Streamlit version of this project is:
* **`src/main.py`**

### 2. Dependencies (`requirements.txt`)
Verify that `requirements.txt` is in the root directory. Streamlit Community Cloud will automatically detect this file and install the listed dependencies. Key dependencies included in this project:
* `streamlit>=1.35.0`
* `pandas>=2.0.0`
* `numpy>=1.24.0`
* `datasets>=2.14.0` (for downloading the Hugging Face dataset)
* `groq>=0.12.0` (for LLM recommendation generations)
* `python-dotenv>=1.0.0`
* `fuzzywuzzy>=0.18.0` & `python-Levenshtein>=0.21.0`

### 3. Theme Configuration (`.streamlit/config.toml`)
To keep the custom branding, Outfit + Playfair Display font styles, and premium dark/red theme settings, Streamlit needs the configuration file to be located in the `.streamlit` directory.

> [!IMPORTANT]
> Rename or copy the root `streamlit_config.toml` file to **`.streamlit/config.toml`** in your repository before pushing to GitHub.
>
> **Command to copy:**
> ```bash
> mkdir .streamlit
> cp streamlit_config.toml .streamlit/config.toml
> ```

---

## 🚀 Step 2: Deploy to Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click the **"New app"** button in the upper-right corner.
3. In the deployment form, configure the fields as follows:
   * **Repository:** Select your GitHub repository (e.g., `username/zomato-bot`).
   * **Branch:** Select your main branch (e.g., `main` or `master`).
   * **Main file path:** Set this exactly to **`src/main.py`**.

---

## 🔑 Step 3: Configure Environment Secrets

Your application requires a Groq API Key to run LLM recommendation operations. Rather than committing your key (which is security risk), set it as a Streamlit secret.

1. Click on the **"Advanced settings..."** button at the bottom of the deployment form.
2. Under the **Secrets** text area, paste the following keys in TOML format:

```toml
# Groq LLM API Key (Required)
GROQ_API_KEY = "your_actual_groq_api_key_here"

# Optional Overrides (Use if you want to change default values)
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
HF_DATASET_NAME = "ManikaSaini/zomato-restaurant-recommendation"
LOG_LEVEL = "INFO"
```

3. Click **"Save"**.
4. Click **"Deploy!"**.

---

## 📈 Post-Deployment & Maintenance

### Continuous Integration (CI)
Streamlit Community Cloud automatically listens to changes in your GitHub repository. Whenever you push new changes to the selected branch, Streamlit will rebuild and redeploy the app automatically.

### Checking logs
If the application runs into issues, open the Streamlit Cloud dashboard:
1. Click the **"Manage app"** button in the lower-right corner of your running app interface.
2. Review the live console output to monitor download progress of the Hugging Face dataset, API call durations, and any exceptions.

---

## 🔍 Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **App fails on startup with `ModuleNotFoundError`** | Missing packages in `requirements.txt`. | Ensure any newly introduced Python package is listed in `requirements.txt` and push to GitHub. |
| **`No Groq API key configured` Warning Banner** | The environment variable `GROQ_API_KEY` is empty or incorrectly set. | Check **Settings** -> **Secrets** in the Streamlit Cloud Console and ensure `GROQ_API_KEY` is present and holds a valid token. |
| **Slow initial load** | Downloading the dataset (~148MB) from Hugging Face on cold starts. | Streamlit Cloud caches dataset operations using `@st.cache_data`. The first visitor will experience a 10-20s delay during the dataset load, but subsequent loads/clicks will load instantly from the in-memory cache. |
| **App runs out of memory** | High data caching or excessive concurrent users. | Streamlit Cloud has a resource limit of 1GB RAM. Ensure caching configurations are optimized and old runs are cleared out. |
