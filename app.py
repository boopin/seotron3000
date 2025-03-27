"""
SEOtron 3000: The Galactic Web Analyzer
Version: 3.0
Updated: March 2025
Description: Unleash the power of SEOtron 3000, a hyper-advanced tool forged by xAI to scan the digital cosmos!
Analyze webpages with laser precision—meta tags, headings, links, readability, image SEO, mobile readiness, and more.
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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import io

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

def get_load_time(url, full_render=False, retries=3):
    for attempt in range(retries):
        try:
            if full_render:
                options = Options()
                options.headless = True
                driver = webdriver.Chrome(options=options)
                start_time = time.time()
                driver.get(url)
                end_time = time.time()
                driver.quit()
            else:
                start_time = time.time()
                requests.get(url, timeout=10)
                end_time = time.time()
            return round((end_time - start_time) * 1000)
        except requests.Timeout:
            if attempt == retries - 1:
                return "Timeout Error"
        except requests.ConnectionError:
            return "Connection Error"
        except Exception as e:
            return f"Error: {str(e)}"
    return None

def extract_keywords(text, num_keywords=20, target_keywords=None):
    stop_words = set(stopwords.words("english")) | set(punctuation)
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalnum()]
    word_counts = Counter(words)
    common_keywords = dict(word_counts.most_common(num_keywords))
    if target_keywords:
        densities = {kw: (words.count(kw.lower()) / len(words) * 100) for kw in target_keywords}
        return common_keywords, densities
    return common_keywords, {}

def extract_meta_tags(soup):
    meta_tags = {}
    meta_tags['title'] = soup.title.text.strip() if soup.title else ''
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_tags['description'] = meta_description['content'].strip() if meta_description else ''
    return meta_tags

def extract_headings(soup):
    headings = []
    for i in range(1, 7):
        heading_tag = f'h{i}'
        for h in soup.find_all(heading_tag):
            headings.append({'level': heading_tag.upper(), 'text': h.text.strip()})
    return headings

def extract_internal_links(soup, base_url):
    domain = urlparse(base_url).netloc
    internal_links = []
    for a in soup.find_all('a', href=True):
        href = urljoin(base_url, a['href'])
        if urlparse(href).netloc == domain:
            internal_links.append({'url': href, 'anchor_text': a.text.strip()})
    return internal_links

def extract_external_links(soup, base_url):
    domain = urlparse(base_url).netloc
    external_links = []
    for a in soup.find_all('a', href=True):
        href = urljoin(base_url, a['href'])
        if urlparse(href).netloc != domain:
            external_links.append({'url': href, 'anchor_text': a.text.strip()})
    return external_links

def extract_image_data(soup):
    images = soup.find_all('img')
    image_data = []
    for img in images:
        src = img.get('src', '')
        alt = img.get('alt', '')
        try:
            response = requests.head(urljoin(soup.url, src), timeout=5)
            size = int(response.headers.get('content-length', 0)) / 1024  # KB
        except:
            size = None
        image_data.append({'src': src, 'alt': alt, 'size_kb': size})
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
        response = requests.get(robots_url, timeout=5)
        return "Disallow" in response.text if response.status_code == 200 else "Not Found"
    except:
        return "Error"

def detect_duplicates(contents):
    if len(contents) < 2:
        return None
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def analyze_url(url, full_render=False, target_keywords=None):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    result = {
        'url': url,
        'status': 'Success',
        'load_time_ms': 0,
        'meta_title': '',
        'meta_description': '',
        'h1_count': 0, 'h2_count': 0, 'h3_count': 0, 'h4_count': 0, 'h5_count': 0, 'h6_count': 0,
        'word_count': 0,
        'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0,
        'internal_links': [], 'internal_link_count': 0,
        'external_links': [], 'external_link_count': 0,
        'images': [], 'image_count': 0,
        'mobile_friendly': False,
        'canonical_url': None,
        'robots_txt_status': '',
    }
    try:
        result['load_time_ms'] = get_load_time(url, full_render)
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        soup.url = url  # Attach URL to soup for later use
        text_content = ' '.join([p.text.strip() for p in soup.find_all(['p', 'div', 'span'])])

        result['word_count'] = len(text_content.split())
        result['flesch_reading_ease'] = flesch_reading_ease(text_content)
        result['flesch_kincaid_grade'] = flesch_kincaid_grade(text_content)
        result['gunning_fog'] = gunning_fog(text_content)

        meta_tags = extract_meta_tags(soup)
        result['meta_title'] = meta_tags.get('title', '')
        result['meta_description'] = meta_tags.get('description', '')

        result['headings'] = extract_headings(soup)
        for level in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            result[f'{level.lower()}_count'] = sum(1 for h in result['headings'] if h['level'] == level)

        internal_links = extract_internal_links(soup, url)
        result['internal_links'] = internal_links
        result['internal_link_count'] = len(internal_links)

        external_links = extract_external_links(soup, url)
        result['external_links'] = external_links
        result['external_link_count'] = len(external_links)

        images = extract_image_data(soup)
        result['images'] = images
        result['image_count'] = len(images)

        result['mobile_friendly'] = check_mobile_friendliness(soup)
        result['canonical_url'] = check_canonical(soup)
        result['robots_txt_status'] = check_robots_txt(url)

        keywords, densities = extract_keywords(text_content, target_keywords=target_keywords)
        result['keywords'] = keywords
        result['keyword_densities'] = densities

    except Exception as e:
        result['status'] = f"Error: {str(e)}"

    return result, text_content

def display_wordcloud(keywords):
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(keywords)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def main():
    st.set_page_config(page_title="SEOtron 3000: The Galactic Web Analyzer", layout="wide", page_icon="icon.png")
    st.title("SEOtron 3000: The Galactic Web Analyzer")
    st.markdown("*Scanning the digital cosmos with laser precision!*")

    # Settings
    with st.sidebar:
        st.header("Control Panel")
        full_render = st.checkbox("Full Render Load Time (Selenium)", False)
        target_keywords = st.text_input("Target Keywords (comma-separated)", "").split(",")
        target_keywords = [kw.strip() for kw in target_keywords if kw.strip()] or None

    # Input URLs
    urls_input = st.text_area("Enter URLs (one per line, max 10)", height=200)
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

    if st.button("Launch Analysis"):
        if urls:
            urls = [preprocess_url(url) for url in urls]
            if len(urls) > 10:
                st.warning("Max 10 URLs allowed. Analyzing first 10.")
                urls = urls[:10]

            results = []
            contents = []
            internal_links_data = []
            external_links_data = []
            headings_data = []
            images_data = []
            progress_bar = st.progress(0)

            with st.spinner("Engaging hyperdrive..."):
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_url = {executor.submit(analyze_url, url, full_render, target_keywords): url for url in urls}
                    for i, future in enumerate(future_to_url):
                        result, content = future.result()
                        results.append(result)
                        contents.append(content)
                        progress_bar.progress((i + 1) / len(urls))

                        for link in result.get('internal_links', []):
                            internal_links_data.append({'page_url': result['url'], 'link_url': link['url'], 'anchor_text': link['anchor_text']})
                        for link in result.get('external_links', []):
                            external_links_data.append({'page_url': result['url'], 'link_url': link['url'], 'anchor_text': link['anchor_text']})
                        for heading in result['headings']:
                            headings_data.append({'page_url': result['url'], 'level': heading['level'], 'text': heading['text']})
                        for img in result.get('images', []):
                            images_data.append({'page_url': result['url'], 'src': img['src'], 'alt': img['alt'], 'size_kb': img['size_kb']})

            df = pd.DataFrame(results)
            duplicate_matrix = detect_duplicates(contents)

            # Tabs
            tabs = st.tabs(["Summary", "Main Table", "Internal Links", "External Links", "Headings", "Images", "Visualizations"])

            with tabs[0]:
                st.subheader("Galactic Summary")
                summary = {
                    "Avg Load Time (ms)": [df['load_time_ms'].mean()],
                    "Avg Word Count": [df['word_count'].mean()],
                    "Avg Internal Links": [df['internal_link_count'].mean()],
                    "Avg External Links": [df['external_link_count'].mean()],
                    "Avg Flesch Reading Ease": [df['flesch_reading_ease'].mean()],
                    "Avg Flesch-Kincaid Grade": [df['flesch_kincaid_grade'].mean()],
                    "Avg Gunning Fog": [df['gunning_fog'].mean()],
                    "Avg Image Count": [df['image_count'].mean()],
                    "Mobile-Friendly Sites": [df['mobile_friendly'].sum()],
                    "Total URLs": [len(df)],
                }
                st.dataframe(pd.DataFrame(summary))
                if duplicate_matrix is not None:
                    st.write("Duplicate Content Similarity (Cosine):")
                    st.write(duplicate_matrix)

                # Simplified Readability Legend
                st.markdown("### Readability Legend")
                st.markdown("""
                **Flesch Reading Ease (0-100):** Measures text readability. Higher scores indicate easier reading.  
                - 70-100: Very easy to read.  
                - 50-70: Moderately easy, suitable for most audiences.  
                - 0-50: Difficult to read.  

                **Flesch-Kincaid Grade (Grade Level):** Indicates the U.S. grade level needed to understand the text.  
                - 5-7: Easy, readable by 5th-7th graders.  
                - 8-10: Average, suitable for general web content.  
                - 11+: Advanced, requires higher education.  

                **Gunning Fog Index (Grade Level):** Estimates years of education needed based on complex words and sentence length.  
                - 6-8: Easy, widely accessible.  
                - 9-12: Moderate, professional-level reading.  
                - 13+: Complex, technical or academic.
                """)

            with tabs[1]:
                st.subheader("Core Metrics")
                display_columns = [
                    'url', 'status', 'load_time_ms', 'word_count', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                    'internal_link_count', 'external_link_count', 'image_count', 'mobile_friendly', 'canonical_url', 'robots_txt_status',
                    'meta_title', 'meta_description', 'h1_count', 'h2_count', 'h3_count', 'h4_count', 'h5_count', 'h6_count'
                ]
                st.download_button("Download Core Metrics", df[display_columns].to_csv(index=False).encode('utf-8'), "core_metrics.csv", "text/csv")
                st.dataframe(df[display_columns])

            with tabs[2]:
                st.subheader("Internal Hyperlinks")
                internal_links_df = pd.DataFrame(internal_links_data)
                st.download_button("Download Internal Links", internal_links_df.to_csv(index=False).encode('utf-8'), "internal_links.csv", "text/csv")
                st.dataframe(internal_links_df)

            with tabs[3]:
                st.subheader("External Hyperlinks")
                external_links_df = pd.DataFrame(external_links_data)
                st.download_button("Download External Links", external_links_df.to_csv(index=False).encode('utf-8'), "external_links.csv", "text/csv")
                st.dataframe(external_links_df)

            with tabs[4]:
                st.subheader("Headings (H1-H6)")
                headings_df = pd.DataFrame(headings_data)
                st.download_button("Download Headings", headings_df.to_csv(index=False).encode('utf-8'), "headings.csv", "text/csv")
                st.dataframe(headings_df)

            with tabs[5]:
                st.subheader("Image SEO Scan")
                images_df = pd.DataFrame(images_data)
                st.download_button("Download Image Data", images_df.to_csv(index=False).encode('utf-8'), "images.csv", "text/csv")
                st.dataframe(images_df)

            with tabs[6]:
                st.subheader("Cosmic Visualizations")
                if not df.empty:
                    st.write("Keyword Cloud (Top Site):")
                    st.pyplot(display_wordcloud(results[0]['keywords']))
                    st.write("Heading Distribution (All Sites):")
                    heading_counts = df[['h1_count', 'h2_count', 'h3_count', 'h4_count', 'h5_count', 'h6_count']].sum()
                    plt.figure(figsize=(10, 5))
                    heading_counts.plot(kind='bar')
                    plt.title("Heading Distribution")
                    st.pyplot(plt)

if __name__ == "__main__":
    main()
