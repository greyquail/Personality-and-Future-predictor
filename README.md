# Multiverse Future Predictor

This is an advanced, interactive **Streamlit** web application that predicts multiple possible futures (a "multiverse" approach) based on your **past decision** and **current mindset**.

## Features
- **Naive Bayes Model** to match past decisions to base outcomes
- **Sentiment Analysis** using TextBlob to adjust probabilities
- **Multiple possible futures** for a fun, "multiverse" feel
- **Interactive Web UI** via Streamlit

## Installation

1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/multiverse-future-predictor.git
   cd multiverse-future-predictor
2.	Install Dependencies:
    pip install -r requirements.txt
3.  Run the App:
    streamlit run script.py
4.	Open the Browser:
	•	Typically, Streamlit runs at http://localhost:8501.

Deploying

Deploy on Streamlit Cloud
	1.	Push your code to a public GitHub repo.
	2.	Go to Streamlit Cloud, sign in with GitHub.
	3.	Deploy a new app → Select your repo → script.py as the entry point.
	4.	Get your shareable URL!

Deploy on Render
	1.	Create a new web service.
	2.	Link your GitHub repo.
	3.	In Build Command: pip install -r requirements.txt
	4.	In Start Command: streamlit run script.py --server.port $PORT
	5.	Deploy → You’ll get a public URL.

Deploy on a VPS (AWS, DigitalOcean, etc.)
	1.	SSH into your server.
	2.	Install Python 3 & pip.
	3.	Clone your repo.
	4.	pip install -r requirements.txt
	5.	nohup streamlit run script.py --server.port 80 --server.address 0.0.0.0 &
	6.	Access via http://your-server-ip/.

Disclaimer


This app is for entertainment purposes only. It uses simple ML + random probability, not a true future prediction system. For a real advanced approach, you’d need far more data, domain-specific models, and real-time updates.
Enjoy your journey through the multiverse!
