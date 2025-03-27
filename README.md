# SEOtron 3000: The Galactic Web Analyzer

![SEOtron 3000 Banner](https://via.placeholder.com/800x200.png?text=SEOtron+3000)  
*Unleash the power of SEOtron 3000, a hyper-advanced tool forged by xAI to scan the digital cosmos! Analyze webpages with laser precision—meta tags, headings, links, readability, image SEO, mobile readiness, and more. Boldly optimize where no site has optimized before!*

**Version**: 3.0  
**Updated**: March 2025  
**Built by**: xAI

---

## Overview

SEOtron 3000 is a Streamlit-based web application designed to analyze SEO metrics for webpages with unparalleled depth and flair. Whether you're a digital explorer or a seasoned SEO commander, this tool equips you with the data needed to conquer search engine rankings.

---

## Features

- **Load Time Analysis**: Measures page load time (simple or full render with Selenium).
- **Meta Tag Extraction**: Pulls meta titles and descriptions.
- **Headings Analysis**: Extracts and counts H1-H6 headings with distribution visuals.
- **Word Count**: Calculates total and average word counts.
- **Link Analysis**: Extracts internal and external links with anchor texts and counts.
- **Readability Scores**: Computes Flesch Reading Ease, Flesch-Kincaid Grade, and Gunning Fog.
- **Keyword Extraction**: Identifies top 20 keywords and target keyword densities.
- **Image SEO**: Analyzes image alt texts, sizes, and counts.
- **Mobile-Friendliness**: Checks for viewport meta tag.
- **Duplicate Content Detection**: Flags similar content across URLs using cosine similarity.
- **Canonical & Robots.txt Checks**: Verifies canonical URLs and crawl directives.
- **Visualizations**: Displays keyword word clouds and heading distributions.
- **Exportable Data**: Downloads results as CSV files for all analyses.
- **Multithreaded Processing**: Analyzes up to 10 URLs concurrently for speed.

---

## Installation

### Prerequisites
- Python 3.8+
- Chrome browser (for full render mode with Selenium)
- Git

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/seotron-3000.git
   cd seotron-3000

2. Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies:
pip install -r requirements.txt

4. Install ChromeDriver (for full render mode):
Download ChromeDriver matching your Chrome version.
Add it to your system PATH or place it in the project directory.

5. Run the application
streamlit run seotron_3000.py

## Usage
1. Launch SEOtron 3000:
After running the command above, the app opens in your default browser at http://localhost:8501.
2. Input URLs:
Enter up to 10 URLs (one per line) in the text area.
3. Configure Settings (Sidebar):
Check "Full Render Load Time" for Selenium-based analysis (requires ChromeDriver).
Add comma-separated target keywords to analyze their densities.
4. Analyze:
Click "Launch Analysis" to start the scan. A progress bar tracks the process.
5. Explore Results:
Navigate tabs: Summary, Main Table, Internal Links, External Links, Headings, Images, Visualizations.
6. Download CSV files for each section.
View keyword clouds and heading charts in the Visualizations tab.

## Contributing
We welcome contributions from across the galaxy! To contribute:

## Fork the repository.
Create a feature branch (git checkout -b feature/awesome-improvement).
Commit your changes (git commit -m "Add awesome improvement").
Push to the branch (git push origin feature/awesome-improvement).
Open a pull request.

## Troubleshooting
Selenium Errors: Ensure ChromeDriver is installed and matches your Chrome version.
Timeout Issues: Increase the timeout in requests.get() or check your network.
Missing NLTK Data: Run the app once—it auto-downloads punkt and stopwords.
License
This project is licensed under the MIT License—see the  file for details.

Acknowledgments
Built with love by xAI.
Powered by Streamlit, BeautifulSoup, Selenium, and more.
Inspired by the infinite possibilities of the digital universe.
