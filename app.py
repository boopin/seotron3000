"""
SEOtron 3000: Advanced Web Analysis Solution
Version: 3.0
Updated: March 2025
Description: A comprehensive SEO analysis tool by xAI, scanning webpages for meta tags, headings, links, readability, image SEO, mobile readiness, and more.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, urljoin
import time
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px

# Set page config as the first Streamlit command
st.set_page_config(page_title="SEOtron 3000: Advanced Web Analysis Solution", layout="wide", page_icon="icon.png")

# Custom CSS for UI/UX improvements
st.markdown("""
    <style>
    /* General Styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #F8F9FA;
        color: #212529;
    }
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E90FF;
        font-weight: 600;
    }
    p {
        color: #212529;
    }
    /* Sidebar Styling */
    .css-1d391kg {  /* Sidebar */
        background-color: #E9ECEF;
        padding: 20px;
    }
    .css-1d391kg h1, .css-1d391kg h2 {
        color: #1E90FF;
    }
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #6F42C1, #1E90FF);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #5A32A3, #0D6EFD);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E9ECEF;
        color: #212529;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E90FF;
        color: white;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #D1E7FF;
        color: #212529;
    }
    /* Table Styling for Custom HTML Tables */
    .custom-table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
    }
    .custom-table th {
        background-color: #1E90FF;
        color: white;
        padding: 10px;
        font-weight: 600;
        text-align: left;
    }
    .custom-table td {
        padding: 8px;
        border-bottom: 1px solid #D1E7FF;
    }
    .custom-table tr:nth-child(even) {
        background-color: #F1F3F5;
    }
    .custom-table tr:hover {
        background-color: #D1E7FF;
    }
    /* Status Badges */
    .badge {
        padding: 5px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .badge-green {
        background-color: #28A745;
        color: white;
    }
    .badge-orange {
        background-color: #FD7E14;
        color: white;
    }
    .badge-red {
        background-color: #DC3545;
        color: white;
    }
    /* Expander Styling */
    .stExpander {
        border: 1px solid #D1E7FF;
        border-radius: 8px;
        background-color: #FFFFFF;
    }
    /* Input Styling */
    .stTextArea textarea, .stTextInput input {
        border: 1px solid #1E90FF;
        border-radius: 8px;
        padding: 10px;
    }
    /* DataFrame Table Styling (for other tabs) */
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th {
        background-color: #1E90FF;
        color: white;
        padding: 10px;
        font-weight: 600;
    }
    .stDataFrame tr:nth-child(even) {
        background-color: #F1F3F5;
    }
    .stDataFrame tr:hover {
        background-color: #D1E7FF;
    }
    .stDataFrame td {
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Pre-download NLTK data
nltk_data_path = "./nltk_data"
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

# Utility Functions
def preprocess_url(url):
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

def get_load_time(url, retries=2):
    for attempt in range(retries):
        try:
            start_time = time.time()
            requests.get(url, timeout=5)
            end_time = time.time()
            return round((end_time - start_time) * 1000)
        except requests.Timeout:
            if attempt == retries - 1:
                return "Timeout after 5s"
        except requests.ConnectionError:
            return "Connection Failed"
        except Exception as e:
            return f"Error: {str(e)}"
    return None

def extract_keywords(text, num_keywords=10, target_keywords=None):
    stop_words = set(stopwords.words("english")) | set(punctuation)
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalnum()]
    word_counts = Counter(words)
    common_keywords = dict(word_counts.most_common(num_keywords))
    if target_keywords:
        densities = {kw: (words.count(kw.lower()) / len(words) * 100) if words else 0 for kw in target_keywords}
        return common_keywords, densities
    return common_keywords, {}

def extract_meta_tags(soup):
    meta_tags = {}
    meta_tags['title'] = soup.title.text.strip() if soup.title else ''
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_tags['description'] = meta_description['content'].strip() if meta_description else ''
    return meta_tags

def analyze_meta_tags(meta_title, meta_description, target_keywords=None):
    # Initialize results
    title_status = "Optimal"
    description_status = "Optimal"
    title_recommendation = []
    description_recommendation = []

    # Meta Title Analysis
    title_length = len(meta_title)
    if not meta_title:
        title_status = "Missing"
        title_recommendation.append("Add a meta title (50-60 characters).")
    else:
        if title_length < 50:
            title_status = "Too Short"
            title_recommendation.append(f"Extend meta title to 50-60 characters (current: {title_length}).")
        elif title_length > 60:
            title_status = "Too Long"
            title_recommendation.append(f"Shorten meta title to 60 characters or less (current: {title_length}).")

        # Check for target keyword in title
        if target_keywords:
            title_lower = meta_title.lower()
            keywords_in_title = [kw for kw in target_keywords if kw.lower() in title_lower]
            if not keywords_in_title:
                title_recommendation.append(f"Include target keyword(s) {', '.join(target_keywords)} in meta title.")

    # Meta Description Analysis
    description_length = len(meta_description)
    if not meta_description:
        description_status = "Missing"
        description_recommendation.append("Add a meta description (150-160 characters).")
    else:
        if description_length < 150:
            description_status = "Too Short"
            description_recommendation.append(f"Extend meta description to 150-160 characters (current: {description_length}).")
        elif description_length > 160:
            description_status = "Too Long"
            description_recommendation.append(f"Shorten meta description to 160 characters or less (current: {description_length}).")

        # Check for target keyword in description
        if target_keywords:
            description_lower = meta_description.lower()
            keywords_in_description = [kw for kw in target_keywords if kw.lower() in description_lower]
            if not keywords_in_description:
                description_recommendation.append(f"Include target keyword(s) {', '.join(target_keywords)} in meta description.")

        # Check for call-to-action in description
        cta_phrases = ["learn", "discover", "explore", "find out", "get started", "shop now", "try now"]
        has_cta = any(phrase in description_lower for phrase in cta_phrases)
        if not has_cta:
            description_recommendation.append("Add a call-to-action (e.g., 'Discover Now') to improve click-through rate.")

    return {
        'title_status': title_status,
        'description_status': description_status,
        'title_recommendation': "; ".join(title_recommendation) if title_recommendation else "No changes needed.",
        'description_recommendation': "; ".join(description_recommendation) if description_recommendation else "No changes needed."
    }

def extract_headings(soup):
    headings = []
    for i in range(1, 7):
        heading_tag = f'h{i}'
        for h in soup.find_all(heading_tag):
            headings.append({'level': heading_tag.upper(), 'text': h.text.strip()})
    levels = [int(h['level'][1]) for h in headings]
    hierarchy_issues = any(levels[i] > levels[i+1] + 1 for i in range(len(levels)-1)) if levels else False
    return headings, hierarchy_issues

def analyze_headers(headings, word_count, target_keywords=None):
    # Initialize results
    header_status = "Optimal"
    header_recommendation = []

    # Check for H1 presence
    h1_count = sum(1 for h in headings if h['level'] == 'H1')
    if h1_count == 0:
        header_status = "Missing H1"
        header_recommendation.append("Add an H1 tag to define the main topic of the page.")
    elif h1_count > 1:
        header_status = "Multiple H1s"
        header_recommendation.append("Use only one H1 tag per page for better SEO.")

    # Check if H1 includes target keyword
    if target_keywords and h1_count == 1:
        h1_text = next((h['text'] for h in headings if h['level'] == 'H1'), "").lower()
        keywords_in_h1 = [kw for kw in target_keywords if kw.lower() in h1_text]
        if not keywords_in_h1:
            header_recommendation.append(f"Include target keyword(s) {', '.join(target_keywords)} in the H1 tag.")

    # Check H2 distribution for scannability (at least 1 H2 per 300 words)
    h2_count = sum(1 for h in headings if h['level'] == 'H2')
    words_per_h2 = word_count / (h2_count + 1) if h2_count > 0 else word_count
    if words_per_h2 > 300:
        header_status = "Poor Scannability" if header_status == "Optimal" else header_status
        header_recommendation.append(f"Add more H2 tags for better scannability (current: {h2_count} H2s for {word_count} words; aim for 1 H2 per 300 words).")

    return {
        'header_status': header_status,
        'header_recommendation': "; ".join(header_recommendation) if header_recommendation else "No changes needed."
    }

def extract_internal_links(soup, base_url):
    domain = urlparse(base_url).netloc
    internal_links = []
    for a in soup.find_all('a', href=True)[:20]:
        href = urljoin(base_url, a['href'])
        if urlparse(href).netloc == domain:
            try:
                response = requests.head(href, timeout=3)
                status = response.status_code
            except:
                status = "Error"
            internal_links.append({'url': href, 'anchor_text': a.text.strip(), 'status_code': status})
    return internal_links

def extract_external_links(soup, base_url):
    domain = urlparse(base_url).netloc
    external_links = []
    for a in soup.find_all('a', href=True)[:20]:
        href = urljoin(base_url, a['href'])
        if urlparse(href).netloc and urlparse(href).netloc != domain:
            try:
                response = requests.head(href, timeout=3)
                status = response.status_code
            except:
                status = "Error"
            external_links.append({'url': href, 'anchor_text': a.text.strip(), 'status_code': status})
    return external_links

def extract_image_data(soup):
    images = soup.find_all('img')[:10]
    image_data = []
    for img in images:
        src = img.get('src', '')
        alt = img.get('alt', '')
        has_alt = bool(alt)
        try:
            response = requests.head(urljoin(soup.url, src), timeout=3)
            size = int(response.headers.get('content-length', 0)) / 1024
        except:
            size = None
        image_data.append({'src': src, 'alt': alt, 'size_kb': size, 'has_alt': has_alt})
    return image_data

def check_mobile_friendliness(soup):
    viewport = soup.find('meta', attrs={'name': 'viewport'})
    return bool(viewport and 'width=device-width' in viewport.get('content', ''))

def check_canonical(soup):
    canonical = soup.find('link', attrs={'rel': 'canonical'})
    return canonical['href'] if canonical else None

def check_robots_txt(url):
    domain = urlparse(url).netloc
    robots_url = f"https://{domain}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=3)
        return "Disallow" in response.text if response.status_code == 200 else "Not Found"
    except:
        return "Error"

def check_robots_block(url, robots_txt_status, soup_url_path):
    if robots_txt_status == "Not Found" or robots_txt_status == "Error":
        return "Unknown (robots.txt not accessible)"
    domain = urlparse(url).netloc
    robots_url = f"https://{domain}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=3)
        if response.status_code != 200:
            return "Unknown (robots.txt fetch failed)"
        robots_content = response.text.lower()
        # Simple check for disallow rules affecting the URL path
        for line in robots_content.split('\n'):
            if line.startswith('disallow:'):
                disallowed_path = line.split('disallow:')[1].strip().lower()
                if disallowed_path and soup_url_path.lower().startswith(disallowed_path):
                    return "Blocked by robots.txt"
        return "Not Blocked"
    except:
        return "Error"

def check_https(url, soup):
    # Check if the URL uses HTTPS
    protocol = urlparse(url).scheme
    https_status = "Uses HTTPS" if protocol == "https" else "HTTP (Insecure)"
    
    # Check for mixed content if the page uses HTTPS
    mixed_content_issues = []
    if protocol == "https":
        # Look for HTTP resources (images, scripts, stylesheets)
        for tag, attr in [('img', 'src'), ('script', 'src'), ('link', 'href')]:
            for element in soup.find_all(tag):
                resource_url = element.get(attr, '')
                if resource_url and urlparse(resource_url).scheme == 'http':
                    mixed_content_issues.append(f"{tag} with URL: {resource_url}")
    
    if mixed_content_issues:
        https_status += f" | Mixed Content Detected ({len(mixed_content_issues)} issues)"
    return https_status, mixed_content_issues

def check_indexability(soup, url, robots_txt_status):
    # Check for noindex meta tag
    noindex_tag = soup.find('meta', attrs={'name': 'robots', 'content': lambda x: x and 'noindex' in x.lower()})
    if noindex_tag:
        return "Noindex Tag Detected"
    
    # Check if blocked by robots.txt
    soup_url_path = urlparse(url).path
    robots_block_status = check_robots_block(url, robots_txt_status, soup_url_path)
    if "Blocked" in robots_block_status:
        return robots_block_status
    
    # Check for login requirement (e.g., 401 Unauthorized)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 401:
            return "Login Required (401 Unauthorized)"
    except:
        pass  # If request fails, we already handle it in the main analysis
    
    return "Indexable"

def detect_duplicates(contents):
    if len(contents) < 2:
        return None
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except:
        return None

def calculate_seo_score(result):
    score = 0
    if isinstance(result['load_time_ms'], (int, float)):
        score += max(0, 20 - (result['load_time_ms'] / 100))
    score += 20 if result['mobile_friendly'] else 0
    readability = (result['flesch_reading_ease'] / 100) * 20
    score += readability
    link_score = min(result['internal_link_count'] / 10, 1) * 10 + min(result['external_link_count'] / 5, 1) * 10
    score += link_score
    image_score = min(result['image_count'] / 5, 1) * 20 if all(img['has_alt'] for img in result['images']) else 10
    score += image_score
    # Deduct points for HTTPS issues
    if "HTTP (Insecure)" in result['https_status']:
        score -= 10
    if "Mixed Content Detected" in result['https_status']:
        score -= 5
    # Deduct points for indexability issues
    if result['indexability_status'] != "Indexable":
        score -= 15
    # Deduct points for meta tag issues
    if result['meta_title_status'] in ["Missing", "Too Short", "Too Long"]:
        score -= 5
    if result['meta_description_status'] in ["Missing", "Too Short", "Too Long"]:
        score -= 5
    # Deduct points for header issues
    if result['header_status'] in ["Missing H1", "Multiple H1s", "Poor Scannability"]:
        score -= 5
    return min(round(score), 100)

def analyze_url(url, target_keywords=None):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    result = {
        'url': url,
        'status': 'Success',
        'load_time_ms': 0,
        'meta_title': '',
        'meta_description': '',
        'meta_title_status': '',
        'meta_description_status': '',
        'meta_title_recommendation': '',
        'meta_description_recommendation': '',
        'header_status': '',
        'header_recommendation': '',
        'h1_count': 0, 'h2_count': 0, 'h3_count': 0, 'h4_count': 0, 'h5_count': 0, 'h6_count': 0,
        'word_count': 0,
        'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0,
        'internal_links': [], 'internal_link_count': 0,
        'external_links': [], 'external_link_count': 0,
        'images': [], 'image_count': 0,
        'mobile_friendly': False,
        'canonical_url': None,
        'robots_txt_status': '',
        'seo_score': 0,
        'hierarchy_issues': False,
        'alt_text_missing': 0,
        'https_status': '',
        'mixed_content_issues': [],
        'indexability_status': ''
    }
    try:
        st.write(f"Fetching {url}...")
        result['load_time_ms'] = get_load_time(url)
        if "Error" in str(result['load_time_ms']):
            raise Exception(result['load_time_ms'])
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        soup.url = url
        text_content = ' '.join([p.text.strip() for p in soup.find_all(['p', 'div', 'span'])])

        result['word_count'] = len(text_content.split())
        result['flesch_reading_ease'] = flesch_reading_ease(text_content)
        result['flesch_kincaid_grade'] = flesch_kincaid_grade(text_content)
        result['gunning_fog'] = gunning_fog(text_content)

        meta_tags = extract_meta_tags(soup)
        result['meta_title'] = meta_tags.get('title', '')
        result['meta_description'] = meta_tags.get('description', '')

        # Meta Tag Analysis
        meta_analysis = analyze_meta_tags(result['meta_title'], result['meta_description'], target_keywords)
        result['meta_title_status'] = meta_analysis['title_status']
        result['meta_description_status'] = meta_analysis['description_status']
        result['meta_title_recommendation'] = meta_analysis['title_recommendation']
        result['meta_description_recommendation'] = meta_analysis['description_recommendation']

        headings, hierarchy_issues = extract_headings(soup)
        result['headings'] = headings
        result['hierarchy_issues'] = hierarchy_issues
        for level in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            result[f'{level.lower()}_count'] = sum(1 for h in headings if h['level'] == level)

        # Header Tag Analysis
        header_analysis = analyze_headers(headings, result['word_count'], target_keywords)
        result['header_status'] = header_analysis['header_status']
        result['header_recommendation'] = header_analysis['header_recommendation']

        internal_links = extract_internal_links(soup, url)
        result['internal_links'] = internal_links
        result['internal_link_count'] = len(internal_links)

        external_links = extract_external_links(soup, url)
        result['external_links'] = external_links
        result['external_link_count'] = len(external_links)

        images = extract_image_data(soup)
        result['images'] = images
        result['image_count'] = len(images)
        result['alt_text_missing'] = sum(1 for img in images if not img['has_alt'])

        result['mobile_friendly'] = check_mobile_friendliness(soup)
        result['canonical_url'] = check_canonical(soup)
        result['robots_txt_status'] = check_robots_txt(url)

        # HTTPS and Security Checks
        https_status, mixed_content_issues = check_https(url, soup)
        result['https_status'] = https_status
        result['mixed_content_issues'] = mixed_content_issues

        # Indexability Check
        result['indexability_status'] = check_indexability(soup, url, result['robots_txt_status'])

        keywords, densities = extract_keywords(text_content, target_keywords=target_keywords)
        result['keywords'] = keywords
        result['keyword_densities'] = densities

        result['seo_score'] = calculate_seo_score(result)

    except Exception as e:
        result['status'] = f"Error: {str(e)}"
        st.error(f"❌ Analysis failed for {url}: {str(e)}")

    return result, text_content

# Styling function for coloring cells in st.dataframe
def color_cells(val, color_map=None):
    if color_map:
        color = color_map.get(val, "black")
        return f"color: {color}; font-weight: bold"
    return None

def color_numerical_cells(val, thresholds, colors):
    for threshold, color in thresholds:
        if val <= threshold:
            return f"color: {color}; font-weight: bold"
    return f"color: {colors[-1]}; font-weight: bold"

# Function to apply badge styling to status columns
def apply_badge(val):
    if val in ["Very Easy", "Easy", "Low", "No Issues", "Excellent", "Optimal", "Uses HTTPS", "Indexable"]:
        return f'<span class="badge badge-green">{val}</span>'
    elif val in ["Moderate", "Average", "Good", "Increase", "Unknown (robots.txt not accessible)", "Too Short"]:
        return f'<span class="badge badge-orange">{val}</span>'
    elif val in ["Difficult", "Advanced", "Complex", "High", "Issues Detected", "Needs Improvement", "Reduce", "HTTP (Insecure)", "Noindex Tag Detected", "Blocked by robots.txt", "Login Required (401 Unauthorized)", "Missing", "Too Long", "Missing H1", "Multiple H1s", "Poor Scannability"]:
        return f'<span class="badge badge-red">{val}</span>'
    return val

# Function to convert DataFrame to HTML table for rendering badges
def df_to_html_table(df):
    # Start the HTML table
    html = '<table class="custom-table">'
    
    # Add header row
    html += '<tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr>'
    
    # Add data rows
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row:
            # If the value is already an HTML string (e.g., a badge), use it directly; otherwise, escape it
            if isinstance(val, str) and val.startswith('<span class="badge'):
                html += f'<td>{val}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    
    # Close the table
    html += '</table>'
    return html

def main():
    # Header with updated tagline
    st.markdown("""
        <h1 style='text-align: center; color: #1E90FF;'>🚀 SEOtron 3000: Advanced Web Analysis Solution</h1>
        <p style='text-align: center; font-style: italic; color: #6F42C1;'>Delivering Precision Insights for Digital Excellence</p>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Control Panel")
        target_keywords = st.text_input(
            "Target Keywords (comma-separated)",
            placeholder="e.g., SEO, web analysis",
            help="Enter keywords to analyze their density on the pages."
        ).split(",")
        target_keywords = [kw.strip() for kw in target_keywords if kw.strip()] or None
        st.markdown("---")
        st.info("📋 Enter up to 5 URLs to analyze. Results will display in tabs below.")

    # Input Section
    st.subheader("🌐 Analyze Websites")
    urls_input = st.text_area(
        "Enter URLs (one per line, max 5)",
        height=150,
        placeholder="https://example.com\nhttps://wikipedia.org",
        help="Enter up to 5 URLs, one per line, to analyze their SEO performance."
    )
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

    if 'results' not in st.session_state:
        st.session_state.results = []

    col1, col2 = st.columns(2)
    with col1:
        launch = st.button("🚀 Launch Analysis", use_container_width=True)
    with col2:
        retry = st.button("🔄 Retry Failed URLs", use_container_width=True)

    if launch or retry:
        if urls or retry:
            if retry:
                urls = [r['url'] for r in st.session_state.results if r['status'] != "Success"]
            urls = [preprocess_url(url) for url in urls]
            if len(urls) > 5:
                st.warning("⚠️ Max 5 URLs allowed. Analyzing first 5.")
                urls = urls[:5]

            results = []
            contents = []
            internal_links_data = []
            external_links_data = []
            headings_data = []
            images_data = []

            with st.spinner("🔍 Analyzing URLs..."):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_url = {executor.submit(analyze_url, url, target_keywords): url for url in urls}
                    for i, future in enumerate(future_to_url):
                        try:
                            result, content = future.result()
                            results.append(result)
                            contents.append(content)
                            st.write(f"✅ Completed {i+1}/{len(urls)}: {result['url']}")
                        except Exception as e:
                            st.error(f"❌ Analysis failed for {future_to_url[future]}: {str(e)}")
                            results.append({'url': future_to_url[future], 'status': f"Error: {str(e)}"})
                            contents.append("")

                        for link in result.get('internal_links', []):
                            internal_links_data.append({'page_url': result['url'], 'link_url': link['url'], 'anchor_text': link['anchor_text'], 'status_code': link['status_code']})
                        for link in result.get('external_links', []):
                            external_links_data.append({'page_url': result['url'], 'link_url': link['url'], 'anchor_text': link['anchor_text'], 'status_code': link['status_code']})
                        for heading in result.get('headings', []):
                            headings_data.append({'page_url': result['url'], 'level': heading['level'], 'text': heading['text']})
                        for img in result.get('images', []):
                            images_data.append({'page_url': result['url'], 'src': img['src'], 'alt': img['alt'], 'size_kb': img['size_kb'], 'has_alt': img['has_alt']})

            st.session_state.results = results
            df = pd.DataFrame(results)

            # Add a column for shortened URLs (domain only) for x-axis labels
            df['short_url'] = df['url'].apply(lambda x: urlparse(x).netloc)

            # Add status columns for readability scores
            df['flesch_reading_ease_status'] = df['flesch_reading_ease'].apply(
                lambda x: "Very Easy" if x >= 70 else "Moderate" if x >= 50 else "Difficult"
            )
            df['flesch_kincaid_grade_status'] = df['flesch_kincaid_grade'].apply(
                lambda x: "Easy" if x <= 7 else "Average" if x <= 10 else "Advanced"
            )
            df['gunning_fog_status'] = df['gunning_fog'].apply(
                lambda x: "Easy" if x <= 8 else "Moderate" if x <= 12 else "Complex"
            )

            duplicate_matrix = detect_duplicates(contents)

            # Tabs
            tabs = st.tabs(["📊 Summary", "📋 Main Table", "📝 Meta Tags", "📑 Header Tags", "🔗 Internal Links", "🌍 External Links", "📑 Headings", "🖼️ Images", "📈 Visual Dashboard"])

            with tabs[0]:
                st.subheader("📊 Analysis Summary")
                
                # Overview Table
                st.markdown("#### 📋 Overview")
                summary_data = {
                    "Metric": ["Avg Load Time (ms)", "Avg Word Count", "Avg Internal Links", "Avg External Links", 
                               "Avg Image Count", "Mobile-Friendly Sites", "Total URLs Analyzed"],
                    "Value": [f"{df['load_time_ms'].mean():.2f}", f"{df['word_count'].mean():.0f}", 
                              f"{df['internal_link_count'].mean():.1f}", f"{df['external_link_count'].mean():.1f}", 
                              f"{df['image_count'].mean():.1f}", f"{df['mobile_friendly'].sum()}", f"{len(df)}"]
                }
                summary_df = pd.DataFrame(summary_data)
                st.markdown(df_to_html_table(summary_df), unsafe_allow_html=True)

                # Readability and SEO Scores Table
                st.markdown("#### 📖 Readability & SEO Scores")
                readability_data = {
                    "Metric": ["Avg Flesch Reading Ease", "Avg Flesch-Kincaid Grade", "Avg Gunning Fog", "Avg SEO Score"],
                    "Score": [f"{df['flesch_reading_ease'].mean():.2f}", f"{df['flesch_kincaid_grade'].mean():.2f}", 
                              f"{df['gunning_fog'].mean():.2f}", f"{df['seo_score'].mean():.2f}"],
                    "Status": [
                        "Very Easy" if df['flesch_reading_ease'].mean() >= 70 else "Moderate" if df['flesch_reading_ease'].mean() >= 50 else "Difficult",
                        "Easy" if df['flesch_kincaid_grade'].mean() <= 7 else "Average" if df['flesch_kincaid_grade'].mean() <= 10 else "Advanced",
                        "Easy" if df['gunning_fog'].mean() <= 8 else "Moderate" if df['gunning_fog'].mean() <= 12 else "Complex",
                        "Excellent" if df['seo_score'].mean() >= 80 else "Good" if df['seo_score'].mean() >= 50 else "Needs Improvement"
                    ]
                }
                readability_df = pd.DataFrame(readability_data)
                readability_df['Status'] = readability_df['Status'].apply(apply_badge)
                st.markdown(df_to_html_table(readability_df), unsafe_allow_html=True)

                # HTTPS and Security Summary
                st.markdown("#### 🔒 HTTPS and Security Summary")
                security_data = []
                for result in results:
                    if result['status'] == "Success":
                        security_data.append({
                            "URL": result['url'],
                            "HTTPS Status": result['https_status'].split(" | ")[0],
                            "Mixed Content Issues": len(result['mixed_content_issues']),
                            "Status": "No Issues" if "Uses HTTPS" in result['https_status'] and not result['mixed_content_issues'] else "Issues Detected"
                        })
                security_df = pd.DataFrame(security_data)
                security_df['HTTPS Status'] = security_df['HTTPS Status'].apply(apply_badge)
                security_df['Status'] = security_df['Status'].apply(apply_badge)
                st.markdown(df_to_html_table(security_df), unsafe_allow_html=True)

                # Indexability Summary
                st.markdown("#### 📇 Indexability Summary")
                indexability_data = []
                for result in results:
                    if result['status'] == "Success":
                        indexability_data.append({
                            "URL": result['url'],
                            "Indexability Status": result['indexability_status'],
                            "Status": "No Issues" if result['indexability_status'] == "Indexable" else "Issues Detected"
                        })
                indexability_df = pd.DataFrame(indexability_data)
                indexability_df['Indexability Status'] = indexability_df['Indexability Status'].apply(apply_badge)
                indexability_df['Status'] = indexability_df['Status'].apply(apply_badge)
                st.markdown(df_to_html_table(indexability_df), unsafe_allow_html=True)

                # Keyword Density Recommendations
                if target_keywords:
                    st.markdown("#### 🔑 Keyword Density Recommendations")
                    keyword_data = []
                    for result in results:
                        if result['status'] == "Success" and result['keyword_densities']:
                            for kw, density in result['keyword_densities'].items():
                                recommendation = "Optimal" if 1 <= density <= 2 else "Increase" if density < 1 else "Reduce"
                                keyword_data.append({
                                    "URL": result['url'],
                                    "Keyword": kw,
                                    "Density": f"{density:.2f}%",
                                    "Recommendation": recommendation
                                })
                    if keyword_data:
                        keyword_df = pd.DataFrame(keyword_data)
                        keyword_df['Density'] = keyword_df['Density'].apply(
                            lambda val: f'<span class="badge {"badge-green" if 1 <= float(val.strip("%")) <= 2 else "badge-orange" if float(val.strip("%")) < 1 else "badge-red"}">{val}</span>'
                        )
                        keyword_df['Recommendation'] = keyword_df['Recommendation'].apply(apply_badge)
                        st.markdown(df_to_html_table(keyword_df), unsafe_allow_html=True)

                # Broken Links
                broken_internals = [link for link in internal_links_data if isinstance(link['status_code'], int) and link['status_code'] >= 400]
                broken_externals = [link for link in external_links_data if isinstance(link['status_code'], int) and link['status_code'] >= 400]
                if broken_internals or broken_externals:
                    st.markdown("#### ⚠️ Broken Links")
                    broken_data = {
                        "Type": ["Internal", "External"],
                        "Count": [len(broken_internals), len(broken_externals)],
                        "Status": ["Issues Detected" if len(broken_internals) > 0 else "No Issues",
                                   "Issues Detected" if len(broken_externals) > 0 else "No Issues"]
                    }
                    broken_df = pd.DataFrame(broken_data)
                    broken_df['Status'] = broken_df['Status'].apply(apply_badge)
                    st.markdown(df_to_html_table(broken_df), unsafe_allow_html=True)

                # Accessibility Summary
                st.markdown("#### ♿ Accessibility Summary")
                accessibility_data = []
                for result in results:
                    if result['status'] == "Success":
                        accessibility_data.append({
                            "URL": result['url'],
                            "Images Missing Alt Text": result['alt_text_missing'],
                            "Heading Issues": "Yes" if result['hierarchy_issues'] else "No",
                            "Status": "Issues Detected" if result['alt_text_missing'] > 0 or result['hierarchy_issues'] else "No Issues"
                        })
                accessibility_df = pd.DataFrame(accessibility_data)
                accessibility_df['Status'] = accessibility_df['Status'].apply(apply_badge)
                st.markdown(df_to_html_table(accessibility_df), unsafe_allow_html=True)

                # Duplicate Content Similarity
                if duplicate_matrix is not None:
                    st.markdown("#### 📑 Duplicate Content Similarity (Cosine)")
                    duplicate_data = []
                    for i in range(len(duplicate_matrix)):
                        for j in range(len(duplicate_matrix[i])):
                            if i < j:
                                similarity = duplicate_matrix[i][j]
                                status = "Low" if similarity < 0.5 else "Moderate" if similarity < 0.8 else "High"
                                duplicate_data.append({
                                    "URL Pair": f"URL {i+1} vs URL {j+1}",
                                    "Similarity": f"{similarity:.2f}",
                                    "Status": status
                                })
                    if duplicate_data:
                        duplicate_df = pd.DataFrame(duplicate_data)
                        duplicate_df['Similarity'] = duplicate_df['Similarity'].apply(
                            lambda val: f'<span class="badge {"badge-green" if float(val) < 0.5 else "badge-orange" if float(val) < 0.8 else "badge-red"}">{val}</span>'
                        )
                        duplicate_df['Status'] = duplicate_df['Status'].apply(apply_badge)
                        st.markdown(df_to_html_table(duplicate_df), unsafe_allow_html=True)

                # Download Button
                st.markdown("---")
                full_report = pd.concat([df, pd.DataFrame(internal_links_data), pd.DataFrame(external_links_data), pd.DataFrame(headings_data), pd.DataFrame(images_data)], axis=1)
                st.download_button("📥 Download Full Report", full_report.to_csv(index=False).encode('utf-8'), "seotron3000_report.csv", "text/csv", use_container_width=True)

                # Legends in Expander
                with st.expander("📜 View Scoring Legends"):
                    st.markdown("##### 📖 Readability Legend")
                    st.markdown("""
                    - **Flesch Reading Ease (0-100):** Higher scores = easier to read.  
                      - 70-100: Very Easy (Green)  
                      - 50-70: Moderate (Orange)  
                      - 0-50: Difficult (Red)  
                    - **Flesch-Kincaid Grade:** U.S. grade level required.  
                      - 5-7: Easy (Green)  
                      - 8-10: Average (Orange)  
                      - 11+: Advanced (Red)  
                    - **Gunning Fog:** Education years needed.  
                      - 6-8: Easy (Green)  
                      - 9-12: Moderate (Orange)  
                      - 13+: Complex (Red)
                    """)
                    st.markdown("##### 📑 Duplicate Content Similarity Legend")
                    st.markdown("""
                    - **Cosine Similarity (0-1):** Higher values = more duplication.  
                      - 0.0-0.5: Low (Green)  
                      - 0.5-0.8: Moderate (Orange)  
                      - 0.8-1.0: High (Red)
                    """)

            with tabs[1]:
                st.subheader("📋 Main Table")
                display_columns = [
                    'url', 'status', 'load_time_ms', 'word_count', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                    'internal_link_count', 'external_link_count', 'image_count', 'mobile_friendly', 'canonical_url', 'robots_txt_status',
                    'meta_title', 'meta_description', 'meta_title_status', 'meta_description_status', 
                    'meta_title_recommendation', 'meta_description_recommendation',
                    'header_status', 'header_recommendation',
                    'h1_count', 'h2_count', 'h3_count', 'h4_count', 'h5_count', 'h6_count', 'seo_score',
                    'flesch_reading_ease_status', 'flesch_kincaid_grade_status', 'gunning_fog_status',
                    'https_status', 'indexability_status'
                ]
                # Prepare the DataFrame for display
                main_df = df[display_columns].copy()
                # Apply badge styling to the status columns
                main_df['flesch_reading_ease_status'] = main_df['flesch_reading_ease_status'].apply(apply_badge)
                main_df['flesch_kincaid_grade_status'] = main_df['flesch_kincaid_grade_status'].apply(apply_badge)
                main_df['gunning_fog_status'] = main_df['gunning_fog_status'].apply(apply_badge)
                main_df['https_status'] = main_df['https_status'].apply(apply_badge)
                main_df['indexability_status'] = main_df['indexability_status'].apply(apply_badge)
                main_df['meta_title_status'] = main_df['meta_title_status'].apply(apply_badge)
                main_df['meta_description_status'] = main_df['meta_description_status'].apply(apply_badge)
                main_df['header_status'] = main_df['header_status'].apply(apply_badge)
                # Apply numerical coloring to readability scores
                def apply_numerical_coloring(df, column, thresholds, colors):
                    for idx, val in df[column].items():
                        for threshold, color in thresholds:
                            if val <= threshold:
                                df.at[idx, column] = f'<span style="color: {color}; font-weight: bold">{val}</span>'
                                break
                        else:
                            df.at[idx, column] = f'<span style="color: {colors[-1]}; font-weight: bold">{val}</span>'
                    return df

                main_df = apply_numerical_coloring(main_df, 'flesch_reading_ease', [(50, "red"), (70, "orange")], ["green"])
                main_df = apply_numerical_coloring(main_df, 'flesch_kincaid_grade', [(7, "green"), (10, "orange")], ["red"])
                main_df = apply_numerical_coloring(main_df, 'gunning_fog', [(8, "green"), (12, "orange")], ["red"])
                # Convert to HTML table and render
                st.markdown(df_to_html_table(main_df), unsafe_allow_html=True)
                st.download_button("📥 Download Core Metrics", df[display_columns].to_csv(index=False).encode('utf-8'), "core_metrics.csv", "text/csv", use_container_width=True)

            with tabs[2]:
                st.subheader("📝 Meta Tag Optimization Recommendations")
                meta_data = []
                for result in results:
                    if result['status'] == "Success":
                        meta_data.append({
                            "URL": result['url'],
                            "Meta Title": result['meta_title'][:50] + "..." if len(result['meta_title']) > 50 else result['meta_title'],
                            "Title Status": result['meta_title_status'],
                            "Title Recommendation": result['meta_title_recommendation'],
                            "Meta Description": result['meta_description'][:50] + "..." if len(result['meta_description']) > 50 else result['meta_description'],
                            "Description Status": result['meta_description_status'],
                            "Description Recommendation": result['meta_description_recommendation'],
                            "Overall Status": "No Issues" if result['meta_title_status'] == "Optimal" and result['meta_description_status'] == "Optimal" else "Needs Improvement"
                        })
                meta_df = pd.DataFrame(meta_data)
                meta_df['Title Status'] = meta_df['Title Status'].apply(apply_badge)
                meta_df['Description Status'] = meta_df['Description Status'].apply(apply_badge)
                meta_df['Overall Status'] = meta_df['Overall Status'].apply(apply_badge)
                st.markdown(df_to_html_table(meta_df), unsafe_allow_html=True)

                # Add Legend
                with st.expander("📜 View Meta Tag Optimization Legend"):
                    st.markdown("""
                    - **Meta Title Length:**  
                      - 50-60 characters: Optimal (Green)  
                      - <50 characters: Too Short (Orange)  
                      - >60 characters: Too Long (Red)  
                      - Missing: Missing (Red)  
                    - **Meta Description Length:**  
                      - 150-160 characters: Optimal (Green)  
                      - <150 characters: Too Short (Orange)  
                      - >160 characters: Too Long (Red)  
                      - Missing: Missing (Red)
                    """)

            with tabs[3]:
                st.subheader("📑 Header Tag Optimization Recommendations")
                header_data = []
                for result in results:
                    if result['status'] == "Success":
                        h1_text = next((h['text'] for h in result['headings'] if h['level'] == 'H1'), "No H1")
                        header_data.append({
                            "URL": result['url'],
                            "H1 Text": h1_text[:50] + "..." if len(h1_text) > 50 else h1_text,
                            "H1 Count": result['h1_count'],
                            "H2 Count": result['h2_count'],
                            "Header Status": result['header_status'],
                            "Recommendation": result['header_recommendation'],
                            "Overall Status": "No Issues" if result['header_status'] == "Optimal" else "Needs Improvement"
                        })
                header_df = pd.DataFrame(header_data)
                header_df['Header Status'] = header_df['Header Status'].apply(apply_badge)
                header_df['Overall Status'] = header_df['Overall Status'].apply(apply_badge)
                st.markdown(df_to_html_table(header_df), unsafe_allow_html=True)

                # Add Legend
                with st.expander("📜 View Header Tag Optimization Legend"):
                    st.markdown("""
                    - **Header Status:**  
                      - Optimal: H1 present, single H1, good H2 distribution (Green)  
                      - Missing H1: No H1 tag found (Red)  
                      - Multiple H1s: More than one H1 tag (Red)  
                      - Poor Scannability: Not enough H2 tags for content length (Red)
                    """)

            with tabs[4]:
                st.subheader("🔗 Internal Links")
                internal_links_df = pd.DataFrame(internal_links_data)
                st.dataframe(internal_links_df, use_container_width=True)
                st.download_button("📥 Download Internal Links", internal_links_df.to_csv(index=False).encode('utf-8'), "internal_links.csv", "text/csv", use_container_width=True)

            with tabs[5]:
                st.subheader("🌍 External Links")
                external_links_df = pd.DataFrame(external_links_data)
                st.dataframe(external_links_df, use_container_width=True)
                st.download_button("📥 Download External Links", external_links_df.to_csv(index=False).encode('utf-8'), "external_links.csv", "text/csv", use_container_width=True)

            with tabs[6]:
                st.subheader("📑 Headings (H1-H6)")
                headings_df = pd.DataFrame(headings_data)
                st.dataframe(headings_df, use_container_width=True)
                st.download_button("📥 Download Headings", headings_df.to_csv(index=False).encode('utf-8'), "headings.csv", "text/csv", use_container_width=True)

            with tabs[7]:
                st.subheader("🖼️ Image SEO Scan")
                images_df = pd.DataFrame(images_data)
                st.dataframe(images_df, use_container_width=True)
                st.download_button("📥 Download Image Data", images_df.to_csv(index=False).encode('utf-8'), "images.csv", "text/csv", use_container_width=True)

            with tabs[8]:
                st.subheader("📈 Visual Dashboard")
                if not df.empty:
                    st.write("📖 Readability Scores Across URLs:")
                    fig = px.bar(
                        df,
                        x='short_url',
                        y=['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog'],
                        title="Readability Metrics",
                        barmode='group',
                        color_discrete_map={
                            'flesch_reading_ease': '#1E90FF',  # Blue
                            'flesch_kincaid_grade': '#FD7E14',  # Orange
                            'gunning_fog': '#6F42C1'  # Purple
                        },
                        labels={
                            'short_url': 'URL',
                            'value': 'Score',
                            'variable': 'Metric'
                        }
                    )
                    fig.update_layout(
                        height=500,
                        margin=dict(r=100),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12),
                        showlegend=True,
                        xaxis_tickangle=45
                    )
                    fig.update_traces(
                        marker=dict(line=dict(color='#212529', width=1)),
                        opacity=0.9
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("🔗 Link Distribution:")
                    link_fig = px.pie(
                        names=['Internal Links', 'External Links'],
                        values=[df['internal_link_count'].sum(), df['external_link_count'].sum()],
                        title="Link Types",
                        color_discrete_sequence=['#20C997', '#6F42C1']  # Teal and Purple
                    )
                    link_fig.update_traces(
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#212529', width=1))
                    )
                    link_fig.update_layout(
                        height=500,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12)
                    )
                    st.plotly_chart(link_fig, use_container_width=True)

                    st.write("📊 SEO Score Distribution:")
                    seo_fig = px.bar(
                        df,
                        x='short_url',
                        y='seo_score',
                        title="SEO Scores by URL",
                        labels={'seo_score': 'SEO Score', 'short_url': 'URL'},
                        color='seo_score',
                        color_continuous_scale=['#DC3545', '#FD7E14', '#20C997', '#28A745'],  # Red, Orange, Teal, Green
                        range_color=[0, 100],
                        custom_data=['url']
                    )
                    seo_fig.update_layout(
                        xaxis_title="URL",
                        yaxis_title="SEO Score",
                        xaxis_tickangle=45,
                        height=600,
                        margin=dict(r=100),
                        yaxis=dict(
                            gridcolor='lightgray',
                            range=[0, 100]
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12),
                        coloraxis_colorbar=dict(
                            title="SEO Score",
                            x=1.05,
                            xanchor="left",
                            yanchor="middle",
                            len=0.5
                        )
                    )
                    seo_fig.update_traces(
                        hovertemplate="<b>URL</b>: %{customdata[0]}<br><b>SEO Score</b>: %{y}<extra></extra>",
                        marker=dict(
                            line=dict(color='#212529', width=1),
                            opacity=0.9
                        )
                    )
                    seo_fig.add_hline(
                        y=50,
                        line_dash="dash",
                        line_color="#DC3545",  # Red
                        annotation_text="Needs Improvement",
                        annotation_position="top left",
                        annotation=dict(
                            x=0,
                            xanchor="left",
                            yref="paper",
                            y=0.95,
                            font=dict(size=12, color="#DC3545")
                        )
                    )
                    seo_fig.add_hline(
                        y=80,
                        line_dash="dash",
                        line_color="#28A745",  # Green
                        annotation_text="Excellent",
                        annotation_position="top left",
                        annotation=dict(
                            x=0,
                            xanchor="left",
                            yref="paper",
                            y=0.90,
                            font=dict(size=12, color="#28A745")
                        )
                    )
                    st.plotly_chart(seo_fig, use_container_width=True)
                    st.markdown("""
                    **How to Interpret the SEO Scores by URL Chart:**
                    - The x-axis shows the domain of each analyzed URL (shortened for readability).
                    - The y-axis shows the SEO score (0-100) for each URL.
                    - **Color Coding:**
                      - **Red (0-50):** Needs Improvement
                      - **Orange (50-80):** Good
                      - **Teal (80-90):** Very Good
                      - **Green (90-100):** Excellent
                    - Dashed lines mark the thresholds: 50 (Needs Improvement) and 80 (Excellent).
                    - **Hover over a bar** to see the full URL and its exact SEO score.
                    - The color scale on the right indicates the SEO score range.
                    """)

if __name__ == "__main__":
    main()
