"""
SEOtron 3000: The Galactic Web Analyzer
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
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import io
import plotly.express as px

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
                return "Timeout after 10s"
        except requests.ConnectionError:
            return "Connection Failed"
        except Exception as e:
            return f"Error: {str(e)}"
    return None

def extract_keywords(text, num_keywords=20, target_keywords=None):
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

def extract_headings(soup):
    headings = []
    for i in range(1, 7):
        heading_tag = f'h{i}'
        for h in soup.find_all(heading_tag):
            headings.append({'level': heading_tag.upper(), 'text': h.text.strip()})
    levels = [int(h['level'][1]) for h in headings]
    hierarchy_issues = any(levels[i] > levels[i+1] + 1 for i in range(len(levels)-1)) if levels else False
    return headings, hierarchy_issues

def extract_internal_links(soup, base_url):
    domain = urlparse(base_url).netloc
    internal_links = []
    for a in soup.find_all('a', href=True):
        href = urljoin(base_url, a['href'])
        if urlparse(href).netloc == domain:
            try:
                response = requests.head(href, timeout=5)
                status = response.status_code
            except:
                status = "Error"
            internal_links.append({'url': href, 'anchor_text': a.text.strip(), 'status_code': status})
    return internal_links

def extract_external_links(soup, base_url):
    domain = urlparse(base_url).netloc
    external_links = []
    for a in soup.find_all('a', href=True):
        href = urljoin(base_url, a['href'])
        if urlparse(href).netloc and urlparse(href).netloc != domain:
            try:
                response = requests.head(href, timeout=5)
                status = response.status_code
            except:
                status = "Error"
            external_links.append({'url': href, 'anchor_text': a.text.strip(), 'status_code': status})
    return external_links

def extract_image_data(soup):
    images = soup.find_all('img')
    image_data = []
    for img in images:
        src = img.get('src', '')
        alt = img.get('alt', '')
        has_alt = bool(alt)
        try:
            response = requests.head(urljoin(soup.url, src), timeout=5)
            size = int(response.headers.get('content-length', 0)) / 1024  # KB
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

def calculate_seo_score(result):
    score = 0
    if isinstance(result['load_time_ms'], (int, float)):
        score += max(0, 20 - (result['load_time_ms'] / 100))  # Max 20
    score += 20 if result['mobile_friendly'] else 0
    readability = (result['flesch_reading_ease'] / 100) * 20  # Scale to 20
    score += readability
    link_score = min(result['internal_link_count'] / 10, 1) * 10 + min(result['external_link_count'] / 5, 1) * 10
    score += link_score
    image_score = min(result['image_count'] / 5, 1) * 20 if all(img['has_alt'] for img in result['images']) else 10
    score += image_score
    return min(round(score), 100)

@st.cache_data
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
        'seo_score': 0,
        'hierarchy_issues': False,
        'alt_text_missing': 0
    }
    try:
        result['load_time_ms'] = get_load_time(url, full_render)
        if "Error" in str(result['load_time_ms']):
            raise Exception(result['load_time_ms'])
        response = requests.get(url, headers=headers, timeout=10)
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

        headings, hierarchy_issues = extract_headings(soup)
        result['headings'] = headings
        result['hierarchy_issues'] = hierarchy_issues
        for level in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            result[f'{level.lower()}_count'] = sum(1 for h in headings if h['level'] == level)

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

        keywords, densities = extract_keywords(text_content, target_keywords=target_keywords)
        result['keywords'] = keywords
        result['keyword_densities'] = densities

        result['seo_score'] = calculate_seo_score(result)

    except Exception as e:
        result['status'] = f"Error: {str(e)}"

    return result, text_content

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

    if 'results' not in st.session_state:
        st.session_state.results = []

    col1, col2 = st.columns(2)
    with col1:
        launch = st.button("Launch Analysis")
    with col2:
        retry = st.button("Retry Failed URLs")

    if launch or retry:
        if urls or retry:
            if retry:
                urls = [r['url'] for r in st.session_state.results if r['status'] != "Success"]
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

            with st.spinner("Analyzing URLs..."):
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_url = {executor.submit(analyze_url, url, full_render, target_keywords): url for url in urls}
                    for i, future in enumerate(future_to_url):
                        result, content = future.result()
                        results.append(result)
                        contents.append(content)
                        progress_bar.progress((i + 1) / len(urls))

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
            duplicate_matrix = detect_duplicates(contents)

            # Tabs
            tabs = st.tabs(["Summary", "Main Table", "Internal Hyperlinks", "External Hyperlinks", "Headings", "Images", "Visual Dashboard"])

            with tabs[0]:
                st.subheader("Summary")
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
                    "Avg SEO Score": [df['seo_score'].mean()],
                    "Total URLs": [len(df)],
                }
                summary_df = pd.DataFrame(summary)

                # Color-code readability scores and SEO score
                flesch_score = summary_df["Avg Flesch Reading Ease"][0]
                flesch_color = "green" if flesch_score >= 70 else "yellow" if flesch_score >= 50 else "red"
                kincaid_score = summary_df["Avg Flesch-Kincaid Grade"][0]
                kincaid_color = "green" if kincaid_score <= 7 else "yellow" if kincaid_score <= 10 else "red"
                gunning_score = summary_df["Avg Gunning Fog"][0]
                gunning_color = "green" if gunning_score <= 8 else "yellow" if gunning_score <= 12 else "red"
                seo_score = summary_df["Avg SEO Score"][0]
                seo_color = "green" if seo_score >= 80 else "yellow" if seo_score >= 50 else "red"

                st.dataframe(summary_df)
                st.markdown(f"""
                - <span style='color:{flesch_color}'>Avg Flesch Reading Ease: {flesch_score:.2f}</span>  
                - <span style='color:{kincaid_color}'>Avg Flesch-Kincaid Grade: {kincaid_score:.2f}</span>  
                - <span style='color:{gunning_color}'>Avg Gunning Fog: {gunning_score:.2f}</span>  
                - <span style='color:{seo_color}'>Avg SEO Score: {seo_score:.2f}</span>
                """, unsafe_allow_html=True)

                # Keyword Density Recommendations
                if target_keywords:
                    st.write("Keyword Density Recommendations:")
                    for result in results:
                        if result['status'] == "Success" and result['keyword_densities']:
                            st.write(f"URL: {result['url']}")
                            for kw, density in result['keyword_densities'].items():
                                recommendation = "Optimal" if 1 <= density <= 2 else "Increase" if density < 1 else "Reduce"
                                st.write(f"- '{kw}': {density:.2f}% ({recommendation})")

                # Broken Links Summary
                broken_internals = [link for link in internal_links_data if isinstance(link['status_code'], int) and link['status_code'] >= 400]
                broken_externals = [link for link in external_links_data if isinstance(link['status_code'], int) and link['status_code'] >= 400]
                if broken_internals or broken_externals:
                    st.write("**Broken Links Detected:**")
                    st.write(f"Internal: {len(broken_internals)}, External: {len(broken_externals)}")

                # Accessibility Audit
                st.write("**Accessibility Summary:**")
                for result in results:
                    if result['status'] == "Success":
                        st.write(f"URL: {result['url']}")
                        st.write(f"- Images missing alt text: {result['alt_text_missing']}")
                        st.write(f"- Heading hierarchy issues: {'Yes' if result['hierarchy_issues'] else 'No'}")

                # Color-code duplicate content similarity
                if duplicate_matrix is not None:
                    st.markdown("**Duplicate Content Similarity (Cosine):**")
                    for i in range(len(duplicate_matrix)):
                        for j in range(len(duplicate_matrix[i])):
                            if i < j:
                                similarity = duplicate_matrix[i][j]
                                color = "green" if similarity < 0.5 else "yellow" if similarity < 0.8 else "red"
                                st.markdown(f"URLs {i+1} vs {j+1}: <span style='color:{color}'>{similarity:.2f}</span>", unsafe_allow_html=True)

                # Download Full Report
                full_report = pd.concat([summary_df, df, pd.DataFrame(internal_links_data), pd.DataFrame(external_links_data), pd.DataFrame(headings_data), pd.DataFrame(images_data)], axis=1)
                st.download_button("Download Full Report", full_report.to_csv(index=False).encode('utf-8'), "seotron3000_report.csv", "text/csv")

                # Legends
                st.markdown("### Readability Legend")
                st.markdown("""
                **Flesch Reading Ease (0-100):** Measures text readability. Higher scores indicate easier reading.  
                - 70-100: Very easy to read (Green).  
                - 50-70: Moderately easy, suitable for most audiences (Yellow).  
                - 0-50: Difficult to read (Red).  

                **Flesch-Kincaid Grade (Grade Level):** Indicates the U.S. grade level needed to understand the text.  
                - 5-7: Easy, readable by 5th-7th graders (Green).  
                - 8-10: Average, suitable for general web content (Yellow).  
                - 11+: Advanced, requires higher education (Red).  

                **Gunning Fog Index (Grade Level):** Estimates years of education needed based on complex words and sentence length.  
                - 6-8: Easy, widely accessible (Green).  
                - 9-12: Moderate, professional-level reading (Yellow).  
                - 13+: Complex, technical or academic (Red).
                """)

                st.markdown("### Duplicate Content Similarity Legend")
                st.markdown("""
                **Duplicate Content Similarity (Cosine, 0-1):** Measures text similarity between URLs. Higher values indicate more duplication.  
                - 0.0-0.5: Low similarity, unique content (Green).  
                - 0.5-0.8: Moderate similarity, some overlap (Yellow).  
                - 0.8-1.0: High similarity, potential duplicates (Red).
                """)

            with tabs[1]:
                st.subheader("Main Table")
                display_columns = [
                    'url', 'status', 'load_time_ms', 'word_count', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                    'internal_link_count', 'external_link_count', 'image_count', 'mobile_friendly', 'canonical_url', 'robots_txt_status',
                    'meta_title', 'meta_description', 'h1_count', 'h2_count', 'h3_count', 'h4_count', 'h5_count', 'h6_count', 'seo_score'
                ]
                st.dataframe(df[display_columns])
                st.download_button("Download Core Metrics", df[display_columns].to_csv(index=False).encode('utf-8'), "core_metrics.csv", "text/csv")

            with tabs[2]:
                st.subheader("Internal Hyperlinks")
                internal_links_df = pd.DataFrame(internal_links_data)
                st.dataframe(internal_links_df)
                st.download_button("Download Internal Links", internal_links_df.to_csv(index=False).encode('utf-8'), "internal_links.csv", "text/csv")

            with tabs[3]:
                st.subheader("External Hyperlinks")
                external_links_df = pd.DataFrame(external_links_data)
                st.dataframe(external_links_df)
                st.download_button("Download External Links", external_links_df.to_csv(index=False).encode('utf-8'), "external_links.csv", "text/csv")

            with tabs[4]:
                st.subheader("Headings (H1-H6)")
                headings_df = pd.DataFrame(headings_data)
                st.dataframe(headings_df)
                st.download_button("Download Headings", headings_df.to_csv(index=False).encode('utf-8'), "headings.csv", "text/csv")

            with tabs[5]:
                st.subheader("Image SEO Scan")
                images_df = pd.DataFrame(images_data)
                st.dataframe(images_df)
                st.download_button("Download Image Data", images_df.to_csv(index=False).encode('utf-8'), "images.csv", "text/csv")

            with tabs[6]:
                st.subheader("Visual Dashboard")
                if not df.empty:
                    st.write("Readability Scores Across URLs:")
                    fig = px.bar(df, x='url', y=['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog'], title="Readability Metrics", barmode='group')
                    st.plotly_chart(fig)
                    st.write("Link Distribution:")
                    link_fig = px.pie(names=['Internal Links', 'External Links'], values=[df['internal_link_count'].sum(), df['external_link_count'].sum()], title="Link Types")
                    st.plotly_chart(link_fig)
                    st.write("SEO Score Distribution:")
                    seo_fig = px.histogram(df, x='seo_score', title="SEO Score Histogram")
                    st.plotly_chart(seo_fig)

if __name__ == "__main__":
    main()
