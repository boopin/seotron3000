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
    return min(round(score), 100)

def analyze_url(url, target_keywords=None):
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
        st.error(f"Failed to analyze {url}: {str(e)}")

    return result, text_content

def main():
    st.set_page_config(page_title="SEOtron 3000: The Galactic Web Analyzer", layout="wide", page_icon="icon.png")
    
    # Header with styling
    st.markdown("""
        <h1 style='text-align: center; color: #1E90FF;'>SEOtron 3000: The Galactic Web Analyzer</h1>
        <p style='text-align: center; font-style: italic;'>Scanning the digital cosmos with laser precision</p>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        target_keywords = st.text_input("Target Keywords (comma-separated)", placeholder="e.g., SEO, web analysis").split(",")
        target_keywords = [kw.strip() for kw in target_keywords if kw.strip()] or None
        st.markdown("---")
        st.info("Enter up to 5 URLs to analyze. Results will display in tabs below.")

    # Input Section
    st.subheader("Analyze Websites")
    urls_input = st.text_area("Enter URLs (one per line, max 5)", height=150, placeholder="https://example.com\nhttps://wikipedia.org")
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

    if 'results' not in st.session_state:
        st.session_state.results = []

    col1, col2 = st.columns(2)
    with col1:
        launch = st.button("Launch Analysis", use_container_width=True)
    with col2:
        retry = st.button("Retry Failed URLs", use_container_width=True)

    if launch or retry:
        if urls or retry:
            if retry:
                urls = [r['url'] for r in st.session_state.results if r['status'] != "Success"]
            urls = [preprocess_url(url) for url in urls]
            if len(urls) > 5:
                st.warning("Max 5 URLs allowed. Analyzing first 5.")
                urls = urls[:5]

            results = []
            contents = []
            internal_links_data = []
            external_links_data = []
            headings_data = []
            images_data = []

            with st.spinner("Analyzing URLs..."):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_url = {executor.submit(analyze_url, url, target_keywords): url for url in urls}
                    for i, future in enumerate(future_to_url):
                        try:
                            result, content = future.result()
                            results.append(result)
                            contents.append(content)
                            st.write(f"Completed {i+1}/{len(urls)}: {result['url']}")
                        except Exception as e:
                            st.error(f"Analysis failed for {future_to_url[future]}: {str(e)}")
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
            duplicate_matrix = detect_duplicates(contents)

            # Tabs
            tabs = st.tabs(["Summary", "Main Table", "Internal Links", "External Links", "Headings", "Images", "Visual Dashboard"])

            with tabs[0]:
                st.subheader("Analysis Summary")
                
                # Overview Table
                st.markdown("#### Overview")
                summary_data = {
                    "Metric": ["Avg Load Time (ms)", "Avg Word Count", "Avg Internal Links", "Avg External Links", 
                               "Avg Image Count", "Mobile-Friendly Sites", "Total URLs Analyzed"],
                    "Value": [f"{df['load_time_ms'].mean():.2f}", f"{df['word_count'].mean():.0f}", 
                              f"{df['internal_link_count'].mean():.1f}", f"{df['external_link_count'].mean():.1f}", 
                              f"{df['image_count'].mean():.1f}", f"{df['mobile_friendly'].sum()}", f"{len(df)}"]
                }
                st.table(pd.DataFrame(summary_data))

                # Readability and SEO Scores Table
                st.markdown("#### Readability & SEO Scores")
                readability_data = {
                    "Metric": ["Avg Flesch Reading Ease", "Avg Flesch-Kincaid Grade", "Avg Gunning Fog", "Avg SEO Score"],
                    "Score": [f"{df['flesch_reading_ease'].mean():.2f}", f"{df['flesch_kincaid_grade'].mean():.2f}", 
                              f"{df['gunning_fog'].mean():.2f}", f"{df['seo_score'].mean():.2f}"],
                    "Status": [
                        "Very Easy" if df['flesch_reading_ease'].mean() >= 70 else "Moderate" if df['flesch_reading_ease'].mean() >= 50 else "Difficult",
                        "Easy" if df['flesch_kincaid_grade'].mean() <= 7 else "Average" if df['flesch_kincaid_grade'].mean() <= 10 else "Advanced",
                        "Easy" if df['gunning_fog'].mean() <= 8 else "Moderate" if df['gunning_fog'].mean() <= 12 else "Complex",
                        "Excellent" if df['seo_score'].mean() >= 80 else "Good" if df['seo_score'].mean() >= 50 else "Needs Improvement"
                    ],
                    "Color": [
                        "green" if df['flesch_reading_ease'].mean() >= 70 else "orange" if df['flesch_reading_ease'].mean() >= 50 else "red",
                        "green" if df['flesch_kincaid_grade'].mean() <= 7 else "orange" if df['flesch_kincaid_grade'].mean() <= 10 else "red",
                        "green" if df['gunning_fog'].mean() <= 8 else "orange" if df['gunning_fog'].mean() <= 12 else "red",
                        "green" if df['seo_score'].mean() >= 80 else "orange" if df['seo_score'].mean() >= 50 else "red"
                    ]
                }
                readability_df = pd.DataFrame(readability_data)
                readability_df['Score'] = readability_df.apply(lambda row: f"<span style='color:{row['Color']}; font-weight:bold'>{row['Score']}</span>", axis=1)
                readability_df['Status'] = readability_df.apply(lambda row: f"<span style='color:{row['Color']}; font-weight:bold'>{row['Status']}</span>", axis=1)
                # Add darker background to the table
                st.markdown(
                    f"""
                    <style>
                    table {{
                        background-color: #D3D3D3;
                        border-collapse: collapse;
                    }}
                    th, td {{
                        padding: 8px;
                        text-align: left;
                        border: 1px solid #A9A9A9;
                    }}
                    </style>
                    {readability_df[['Metric', 'Score', 'Status']].to_html(escape=False)}
                    """,
                    unsafe_allow_html=True
                )

                # Keyword Density Recommendations
                if target_keywords:
                    st.markdown("#### Keyword Density Recommendations")
                    keyword_data = []
                    for result in results:
                        if result['status'] == "Success" and result['keyword_densities']:
                            for kw, density in result['keyword_densities'].items():
                                recommendation = "Optimal" if 1 <= density <= 2 else "Increase" if density < 1 else "Reduce"
                                color = "green" if 1 <= density <= 2 else "orange" if density < 1 else "red"
                                keyword_data.append({
                                    "URL": result['url'],
                                    "Keyword": kw,
                                    "Density": f"<span style='color:{color}; font-weight:bold'>{density:.2f}%</span>",
                                    "Recommendation": f"<span style='color:{color}; font-weight:bold'>{recommendation}</span>"
                                })
                    if keyword_data:
                        st.markdown(
                            f"""
                            <style>
                            table {{
                                background-color: #D3D3D3;
                                border-collapse: collapse;
                            }}
                            th, td {{
                                padding: 8px;
                                text-align: left;
                                border: 1px solid #A9A9A9;
                            }}
                            </style>
                            {pd.DataFrame(keyword_data).to_html(escape=False)}
                            """,
                            unsafe_allow_html=True
                        )

                # Broken Links
                broken_internals = [link for link in internal_links_data if isinstance(link['status_code'], int) and link['status_code'] >= 400]
                broken_externals = [link for link in external_links_data if isinstance(link['status_code'], int) and link['status_code'] >= 400]
                if broken_internals or broken_externals:
                    st.markdown("#### Broken Links")
                    broken_data = {
                        "Type": ["Internal", "External"],
                        "Count": [len(broken_internals), len(broken_externals)],
                        "Status": ["<span style='color:red'>Issues Detected</span>" if len(broken_internals) > 0 else "No Issues",
                                   "<span style='color:red'>Issues Detected</span>" if len(broken_externals) > 0 else "No Issues"]
                    }
                    st.markdown(
                        f"""
                        <style>
                        table {{
                            background-color: #D3D3D3;
                            border-collapse: collapse;
                        }}
                        th, td {{
                            padding: 8px;
                            text-align: left;
                            border: 1px solid #A9A9A9;
                        }}
                        </style>
                        {pd.DataFrame(broken_data).to_html(escape=False)}
                        """,
                        unsafe_allow_html=True
                    )

                # Accessibility Summary
                st.markdown("#### Accessibility Summary")
                accessibility_data = []
                for result in results:
                    if result['status'] == "Success":
                        accessibility_data.append({
                            "URL": result['url'],
                            "Images Missing Alt Text": result['alt_text_missing'],
                            "Heading Issues": "Yes" if result['hierarchy_issues'] else "No",
                            "Status": "<span style='color:red'>Issues Detected</span>" if result['alt_text_missing'] > 0 or result['hierarchy_issues'] else "<span style='color:green'>No Issues</span>"
                        })
                st.markdown(
                    f"""
                    <style>
                    table {{
                        background-color: #D3D3D3;
                        border-collapse: collapse;
                    }}
                    th, td {{
                        padding: 8px;
                        text-align: left;
                        border: 1px solid #A9A9A9;
                    }}
                    </style>
                    {pd.DataFrame(accessibility_data).to_html(escape=False)}
                    """,
                    unsafe_allow_html=True
                )

                # Duplicate Content Similarity
                if duplicate_matrix is not None:
                    st.markdown("#### Duplicate Content Similarity (Cosine)")
                    duplicate_data = []
                    for i in range(len(duplicate_matrix)):
                        for j in range(len(duplicate_matrix[i])):
                            if i < j:
                                similarity = duplicate_matrix[i][j]
                                color = "green" if similarity < 0.5 else "orange" if similarity < 0.8 else "red"
                                duplicate_data.append({
                                    "URL Pair": f"URL {i+1} vs URL {j+1}",
                                    "Similarity": f"<span style='color:{color}; font-weight:bold'>{similarity:.2f}</span>",
                                    "Status": f"<span style='color:{color}; font-weight:bold'>{'Low' if similarity < 0.5 else 'Moderate' if similarity < 0.8 else 'High'}</span>"
                                })
                    if duplicate_data:
                        st.markdown(
                            f"""
                            <style>
                            table {{
                                background-color: #D3D3D3;
                                border-collapse: collapse;
                            }}
                            th, td {{
                                padding: 8px;
                                text-align: left;
                                border: 1px solid #A9A9A9;
                            }}
                            </style>
                            {pd.DataFrame(duplicate_data).to_html(escape=False)}
                            """,
                            unsafe_allow_html=True
                        )

                # Download Button
                st.markdown("---")
                full_report = pd.concat([df, pd.DataFrame(internal_links_data), pd.DataFrame(external_links_data), pd.DataFrame(headings_data), pd.DataFrame(images_data)], axis=1)
                st.download_button("Download Full Report", full_report.to_csv(index=False).encode('utf-8'), "seotron3000_report.csv", "text/csv", use_container_width=True)

                # Legends in Expander
                with st.expander("View Scoring Legends"):
                    st.markdown("##### Readability Legend")
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
                    st.markdown("##### Duplicate Content Similarity Legend")
                    st.markdown("""
                    - **Cosine Similarity (0-1):** Higher values = more duplication.  
                      - 0.0-0.5: Low (Green)  
                      - 0.5-0.8: Moderate (Orange)  
                      - 0.8-1.0: High (Red)
                    """)

            with tabs[1]:
                st.subheader("Main Table")
                display_columns = [
                    'url', 'status', 'load_time_ms', 'word_count', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                    'internal_link_count', 'external_link_count', 'image_count', 'mobile_friendly', 'canonical_url', 'robots_txt_status',
                    'meta_title', 'meta_description', 'h1_count', 'h2_count', 'h3_count', 'h4_count', 'h5_count', 'h6_count', 'seo_score'
                ]
                st.dataframe(df[display_columns], use_container_width=True)
                st.download_button("Download Core Metrics", df[display_columns].to_csv(index=False).encode('utf-8'), "core_metrics.csv", "text/csv", use_container_width=True)

            with tabs[2]:
                st.subheader("Internal Links")
                internal_links_df = pd.DataFrame(internal_links_data)
                st.dataframe(internal_links_df, use_container_width=True)
                st.download_button("Download Internal Links", internal_links_df.to_csv(index=False).encode('utf-8'), "internal_links.csv", "text/csv", use_container_width=True)

            with tabs[3]:
                st.subheader("External Links")
                external_links_df = pd.DataFrame(external_links_data)
                st.dataframe(external_links_df, use_container_width=True)
                st.download_button("Download External Links", external_links_df.to_csv(index=False).encode('utf-8'), "external_links.csv", "text/csv", use_container_width=True)

            with tabs[4]:
                st.subheader("Headings (H1-H6)")
                headings_df = pd.DataFrame(headings_data)
                st.dataframe(headings_df, use_container_width=True)
                st.download_button("Download Headings", headings_df.to_csv(index=False).encode('utf-8'), "headings.csv", "text/csv", use_container_width=True)

            with tabs[5]:
                st.subheader("Image SEO Scan")
                images_df = pd.DataFrame(images_data)
                st.dataframe(images_df, use_container_width=True)
                st.download_button("Download Image Data", images_df.to_csv(index=False).encode('utf-8'), "images.csv", "text/csv", use_container_width=True)

            with tabs[6]:
                st.subheader("Visual Dashboard")
                if not df.empty:
                    st.write("Readability Scores Across URLs:")
                    fig = px.bar(df, x='url', y=['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog'], title="Readability Metrics", barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("Link Distribution:")
                    link_fig = px.pie(names=['Internal Links', 'External Links'], values=[df['internal_link_count'].sum(), df['external_link_count'].sum()], title="Link Types")
                    st.plotly_chart(link_fig, use_container_width=True)
                    st.write("SEO Score Distribution:")
                    seo_fig = px.histogram(df, x='seo_score', title="SEO Score Histogram")
                    st.plotly_chart(seo_fig, use_container_width=True)

if __name__ == "__main__":
    main()
