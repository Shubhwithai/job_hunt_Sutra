import streamlit as st
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

# Try importing SerpAPI, show error if not installed
try:
    from serpapi import GoogleSearch
except ImportError:
    st.error("""
    The required package 'google-search-results' is not installed.
    Please install it using:
    ```
    pip install google-search-results
    ```
    """)
    st.stop()

# --- Constants ---
SUTRA_API_BASE_URL = "https://api.two.ai/v2"
SUTRA_MODEL_NAME = "sutra-v2"

SUPPORTED_LANGUAGES = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil",
    "Telugu", "Kannada", "Malayalam", "Punjabi", "Marathi",
    "Urdu", "Assamese", "Odia", "Sanskrit", "Korean",
    "Japanese", "Arabic", "French", "German", "Spanish",
    "Portuguese", "Russian", "Chinese", "Vietnamese", "Thai",
    "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch",
    "Italian", "Greek", "Hebrew", "Persian", "Swedish",
    "Norwegian", "Danish", "Finnish", "Czech", "Hungarian",
    "Romanian", "Bulgarian", "Croatian", "Serbian", "Slovak",
    "Slovenian", "Estonian", "Latvian", "Lithuanian", "Malay",
    "Tagalog", "Swahili"
]

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual Job Hub",
    page_icon="💼",
    layout="wide"
)

# --- Streaming Callback Handler (currently not used in the primary translation flow) ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None # Not used by Langchain SChain anymore

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# --- Caching for Chat Model ---
@st.cache_resource
def get_sutra_chat_model(api_key):
    """Initializes and returns a cached ChatOpenAI model for Sutra."""
    return ChatOpenAI(
        api_key=api_key,
        base_url=SUTRA_API_BASE_URL,
        model=SUTRA_MODEL_NAME,
        temperature=0.3,  # Lower temperature for more accurate translations
    )

# --- API Interaction Functions ---

def fetch_company_logo(company_name: str, serp_api_key: str):
    """Fetches a company logo using Google Images via SerpAPI."""
    if not company_name:
        return None
    
    image_params = {
        "api_key": serp_api_key,
        "engine": "google_images",
        "q": f"{company_name} company logo",
        "num": 1,
        "safe": "active",
        "tbm": "isch"  # Image search
    }
    try:
        image_search = GoogleSearch(image_params)
        image_results = image_search.get_dict()
        if image_results.get("images_results") and len(image_results["images_results"]) > 0:
            return image_results["images_results"][0].get("original")
    except Exception as e:
        # Silently fail on logo fetch for now, or use st.warning for debugging
        # st.warning(f"Could not fetch logo for {company_name}: {e}")
        pass
    return None

def fetch_jobs(query: str, serp_api_key: str, num_results: int = 20, location: str = None, job_type: str = None):
    """Fetches job listings using SerpAPI."""
    params = {
        "api_key": serp_api_key,
        "engine": "google_jobs",
        "q": query,
        "google_domain": "google.com",
        "hl": "en", # Fetch in English for consistent processing before translation
        "gl": "us", # Use a general country code, location parameter will refine
    }
    if location:
        params["location"] = location
    
    # Example: If job_type is "Full-time", you might use "chips": "employment_type:FULLTIME"
    # The current implementation appends job_type to the query.
    if job_type:
        params["q"] = f"{query} {job_type}"

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        jobs = results.get("jobs_results", [])
        
        limited_jobs = jobs[:num_results]
        
        enhanced_jobs = []
        for job in limited_jobs:
            job_copy = job.copy()
            logo_url = fetch_company_logo(job_copy.get('company_name', ''), serp_api_key)
            if logo_url:
                job_copy['thumbnail'] = logo_url
            enhanced_jobs.append(job_copy)
        
        return enhanced_jobs
    except Exception as e:
        st.error(f"Error fetching jobs from SerpAPI: {e}")
        return []

def _translate_with_sutra(api_key, system_prompt_content: str, user_prompt_content: str, parse_json: bool = False):
    """Helper function to call Sutra LLM for translation or other tasks."""
    try:
        model = get_sutra_chat_model(api_key)
        messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=user_prompt_content)
        ]
        response = model.invoke(messages)
        result_text = response.content.strip()

        if parse_json:
            # Clean potential markdown code fences around JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            return json.loads(result_text)
        return result_text
    except json.JSONDecodeError as e:
        st.warning(f"Sutra LLM: Failed to parse JSON response: {e}. Raw response: '{result_text}'")
        raise  # Re-raise to be handled by the caller
    except Exception as e:
        st.error(f"Sutra LLM Error: {e}")
        raise # Re-raise to be handled by the caller

def translate_query_to_english(query: str, api_key: str):
    """Translates a search query to English using Sutra LLM."""
    system_prompt = """
    You are a professional translator specializing in job search queries. Translate the following search query to English.
    
    Translation Rules:
    1. Keep the translation concise and clear.
    2. Maintain the search intent and job-related terminology.
    3. Preserve any proper nouns (names, places).
    4. Keep any numbers, dates, and measurements.
    5. Ensure the translation is natural and search-friendly.
    6. For technical terms (like "Full Stack", "Developer", etc.), use standard English terminology.
    7. If the query is already in English, return it as is.
    
    Return ONLY the translated query without any explanations or additional text.
    """
    try:
        # st.write(f"Debug - Original query for translation: {query}") # For debugging
        translated_query = _translate_with_sutra(api_key, system_prompt, query)
        # st.write(f"Debug - Translated query: {translated_query}") # For debugging
        return translated_query
    except Exception: # Catch specific exceptions from _translate_with_sutra if needed
        st.warning("Query translation failed. Using original query.")
        return query


def translate_job_item(job: dict, target_language: str, api_key: str):
    """Translates specific fields of a single job item."""
    system_prompt = f"""
    You are a professional translator specializing in job listings translation. Translate the following job content to {target_language}.
    
    Translation Rules:
    1. Translate ONLY these fields: title, company_name, description, location.
    2. For 'title': Keep it concise and job-focused.
    3. For 'company_name': Translate the company name if it has a common translation in {target_language}; otherwise, keep original.
    4. For 'description': Maintain the job requirements and responsibilities context. Truncate if necessary after translation to be around 300-400 characters, summarizing key points.
    5. For 'location': Translate location information (city, country, etc.).
    
    Guidelines:
    - Ensure natural and fluent language.
    - Maintain original meaning and context.
    - Keep technical terms, skills, and requirements largely in their original form (English is common for these) unless a very well-known {target_language} equivalent exists.
    - Preserve numbers, dates, and measurements.
    - Output ONLY the translated fields in this exact JSON format:
    {{
        "title": "translated title",
        "company_name": "translated company name",
        "description": "translated description",
        "location": "translated location"
    }}
    
    Important:
    - Do not add any explanations.
    - Do not modify the JSON structure.
    - Ensure the translation is culturally appropriate for {target_language} speakers.
    """
    
    fields_to_translate = {
        "title": job.get('title', ''),
        "company_name": job.get('company_name', ''),
        "description": job.get('description', '')[:500], # Truncate description before sending
        "location": job.get('location', '')
    }
    job_json_payload = json.dumps(fields_to_translate, ensure_ascii=False)

    try:
        translated_fields = _translate_with_sutra(api_key, system_prompt, job_json_payload, parse_json=True)
        
        # Create a new job dictionary with updated fields
        updated_job = job.copy()
        updated_job['title'] = translated_fields.get('title', job.get('title', ''))
        updated_job['company_name'] = translated_fields.get('company_name', job.get('company_name', ''))
        updated_job['description'] = translated_fields.get('description', job.get('description', ''))
        updated_job['location'] = translated_fields.get('location', job.get('location', ''))
        return updated_job
    except Exception: # Catch exceptions from _translate_with_sutra or JSON parsing
        return job # Return original job on error


# --- UI Display Functions ---
def display_job_card(job, index, container):
    """Displays a single job card in the given container."""
    with container:
        st.markdown(f"""
            <div class="job-card">
                <div style="display: flex; align-items: flex-start; gap: 20px;">
                    <div style="flex: 3;">
                        <h3 class="job-title">{index}. {job.get('title', 'N/A')}</h3>
                        <p class="company-name">🏢 <strong>Company:</strong> {job.get('company_name', 'N/A')}</p>
                        <p class="job-location">📍 <strong>Location:</strong> {job.get('location', 'N/A')}</p>
                        <p class="job-type">⏰ <strong>Type:</strong> {job.get('detected_extensions', {}).get('schedule_type', 'Not specified')}</p>
                        <p class="job-description">{job.get('description', 'N/A')[:300]}...</p>
                        <p><a href="{job.get('job_highlights', [{}])[0].get('link') or job.get('related_links', [{}])[0].get('link') or job.get('share_link', '#')}" class="job-link" target="_blank">🔗 View Job</a></p>
                    </div>
                    <div style="flex: 1;">
                        {f'<div class="image-container"><img src="{job.get("thumbnail")}" alt="Company Logo"></div>' if job.get('thumbnail') else '<div class="image-container"><p style="text-align:center; color: #ccc;">No Logo</p></div>'}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "serp_api_key" not in st.session_state:
    st.session_state.serp_api_key = ""
if "sutra_api_key" not in st.session_state:
    st.session_state.sutra_api_key = ""
if "jobs_data" not in st.session_state:
    st.session_state.jobs_data = [] # Stores raw fetched jobs
if "displayed_jobs" not in st.session_state:
    st.session_state.displayed_jobs = [] # Stores jobs to be displayed (potentially translated)
if "search_query" not in st.session_state:
    st.session_state.search_query = "AI Engineer" # Default search
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"
if "selected_location" not in st.session_state:
    st.session_state.selected_location = "Worldwide"


# --- Sidebar ---
with st.sidebar:
    st.markdown('<h1>💼 Job Hub</h1>', unsafe_allow_html=True)
    
    st.markdown("### API Keys")
    st.markdown("**SUTRA API Key** ([Get it here](https://www.two.ai/sutra/api))")
    sutra_api_key_input = st.text_input(
        "Enter Sutra API Key:", 
        value=st.session_state.sutra_api_key, 
        type="password", 
        key="sutra_api_key_input",
        label_visibility="collapsed"
    )
    if sutra_api_key_input:
        st.session_state.sutra_api_key = sutra_api_key_input

    st.markdown("**SerpAPI Key** ([Get it here](https://serpapi.com/users/sign_in))")
    serp_api_key_input = st.text_input(
        "Enter SerpAPI Key:", 
        value=st.session_state.serp_api_key, 
        type="password", 
        key="serp_api_key_input",
        label_visibility="collapsed"
    )
    if serp_api_key_input:
        st.session_state.serp_api_key = serp_api_key_input

    st.markdown("### Search Settings")
    st.session_state.selected_location = st.selectbox(
        "Location:",
        ["Worldwide", "India", "United States", "Canada", "United Kingdom", "Germany", "France", "Japan", "Australia", "Singapore"],
        index=["Worldwide", "India", "United States", "Canada", "United Kingdom", "Germany", "France", "Japan", "Australia", "Singapore"].index(st.session_state.selected_location)
    )
    
    st.session_state.selected_language = st.selectbox(
        "Display Language:", 
        SUPPORTED_LANGUAGES, 
        index=SUPPORTED_LANGUAGES.index(st.session_state.selected_language)
    )
    
    st.divider()
    st.markdown(f"Displaying jobs in: **{st.session_state.selected_language}**")

    with st.expander("About Job Hub"):
        st.markdown("""
        This app uses:
        - **SerpAPI** to fetch job listings.
        - **Sutra LLM by TWO.AI** to translate job details.
        - **Streamlit** for the interactive web interface.
        
        Search for jobs globally and view them in your preferred language!
        """)

# --- Main Page ---
st.markdown(
    '<h1><img src="https://framerusercontent.com/images/9vH8BcjXKRcC5OrSfkohhSyDgX0.png" width="60" style="vertical-align: middle; margin-right: 10px;"/>Multilingual Job Search<img src="https://pixcap.com/cdn/library/templates/0bb47b92-ac86-457c-99c6-e05a7c0cf4e3/thumbnail/f7ff3dee-c635-43aa-82bf-596fae43744f_transparent_null_400.webp" width="90" height="90" style="vertical-align: middle; margin-left:10px;"/></h1>',
    unsafe_allow_html=True
)

# Search bar
search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    user_search_query = st.text_input(
        "Search for jobs (e.g., 'Software Developer in London', 'マーケティングマネージャー 東京'):", 
        value=st.session_state.search_query,
        key="search_query_input"
    )
    st.session_state.search_query = user_search_query # Update session state continuously

with search_col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

# --- Logic for Searching and Displaying Jobs ---
if search_button:
    if not st.session_state.serp_api_key:
        st.error("Please enter your SerpAPI key in the sidebar to search for jobs.")
    else:
        st.session_state.jobs_data = [] # Clear previous raw results
        st.session_state.displayed_jobs = [] # Clear previous display results
        
        current_query = st.session_state.search_query
        
        # Translate query to English if necessary
        if st.session_state.selected_language != "English":
            if not st.session_state.sutra_api_key:
                st.warning("Sutra API key not provided. Search will be performed using the original query. Translation features will be disabled.")
                english_query = current_query
            else:
                with st.spinner("Translating search query to English..."):
                    english_query = translate_query_to_english(current_query, st.session_state.sutra_api_key)
                if english_query.lower() != current_query.lower(): # Check if translation occurred
                     st.info(f"Searching for (translated from your query): '{english_query}'")
        else:
            english_query = current_query
        
        effective_location = st.session_state.selected_location if st.session_state.selected_location != "Worldwide" else None
        
        with st.spinner(f"Fetching jobs for '{english_query}' in '{st.session_state.selected_location}'..."):
            raw_jobs = fetch_jobs(
                query=english_query,
                serp_api_key=st.session_state.serp_api_key,
                location=effective_location
            )
            
            if raw_jobs:
                st.session_state.jobs_data = raw_jobs
                st.success(f"Found {len(raw_jobs)} jobs related to '{english_query}'.")
            else:
                st.warning(f"No jobs found for '{english_query}'. Try a different search term or broaden the location.")

# Process and display jobs (called after search or if jobs_data exists from previous search)
if st.session_state.jobs_data:
    jobs_to_process = st.session_state.jobs_data
    
    if st.session_state.selected_language != "English":
        if not st.session_state.sutra_api_key:
            if not search_button: # Avoid double warning if search just happened
                 st.warning("Sutra API key needed for translation. Displaying jobs in English.")
            st.session_state.displayed_jobs = jobs_to_process
        else:
            st.session_state.displayed_jobs = [] # Reset for new translations
            with st.spinner(f"Translating {len(jobs_to_process)} jobs to {st.session_state.selected_language}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                translated_job_list = []
                for i, job_item in enumerate(jobs_to_process):
                    status_text.text(f"Translating job {i+1}/{len(jobs_to_process)}...")
                    translated_job = translate_job_item(job_item, st.session_state.selected_language, st.session_state.sutra_api_key)
                    translated_job_list.append(translated_job)
                    progress_bar.progress((i + 1) / len(jobs_to_process))
                
                st.session_state.displayed_jobs = translated_job_list
                status_text.success(f"All {len(jobs_to_process)} jobs processed for {st.session_state.selected_language}.")
                progress_bar.empty() # Remove progress bar after completion
    else:
        st.session_state.displayed_jobs = jobs_to_process

    if st.session_state.displayed_jobs:
        st.markdown(f"--- \n### Displaying Jobs in {st.session_state.selected_language}")
        jobs_display_container = st.container()
        for i, job in enumerate(st.session_state.displayed_jobs):
            display_job_card(job, i + 1, jobs_display_container)
    elif not search_button : # Only show this if not immediately after a search that found nothing
        st.info("No jobs to display. Please perform a search.")

elif not st.session_state.serp_api_key:
    st.info("Welcome to the Multilingual Job Hub! Please enter your API keys in the sidebar and start your job search.")
else:
     st.info("Start by searching for jobs using the search bar above.")


# --- Custom CSS ---
st.markdown("""
    <style>
    /* Theme-aware styles */
    .job-card {
        background-color: var(--background-color); /* Streamlit theme variable */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Slightly more pronounced shadow */
        border: 1px solid var(--border-color); /* Custom variable, defined below */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .job-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .job-title {
        color: var(--text-color); /* Streamlit theme variable */
        font-size: 1.4em; /* Slightly adjusted */
        margin-bottom: 8px;
        font-weight: 600;
    }
    .company-name, .job-location, .job-type {
        color: var(--text-color-secondary); /* Custom variable */
        font-size: 1.05em; /* Slightly adjusted */
        margin: 4px 0;
    }
    .job-type {
        color: var(--accent-color); /* Custom variable */
    }
    .job-description {
        color: var(--text-color-secondary);
        font-size: 0.95em; /* Slightly adjusted */
        margin: 12px 0;
        line-height: 1.6;
    }
    .job-link {
        color: var(--link-color); /* Custom variable */
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        padding: 6px 10px;
        border: 1px solid var(--link-color);
        border-radius: 5px;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .job-link:hover {
        background-color: var(--link-color);
        color: var(--background-color); /* For contrast on hover */
        text-decoration: none;
    }
    .image-container {
        width: 120px; /* Fixed width */
        height: 120px; /* Fixed height for consistency */
        overflow: hidden;
        border-radius: 8px; /* Softer radius */
        border: 1px solid var(--border-color);
        background-color: #ffffff; /* White background for all logos for consistency */
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 8px; /* Padding around logo */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    /* Define custom variables based on Streamlit's theme data attribute */
    body[data-theme="light"] {
        --background-color: #ffffff; /* Streamlit's light theme background */
        --text-color: #31333F; /* Streamlit's light theme text */
        --text-color-secondary: #5A5A5A; /* Darker gray for secondary text */
        --border-color: #E0E0E0; /* Light gray border */
        --accent-color: #007BFF; /* Bootstrap primary blue as an example */
        --link-color: #0062CC; 
    }

    body[data-theme="dark"] {
        --background-color: #0E1117; /* Streamlit's dark theme background */
        --text-color: #FAFAFA; /* Streamlit's dark theme text */
        --text-color-secondary: #A0A0A0; /* Lighter gray for secondary text */
        --border-color: #3E3E3E; /* Darker gray border */
        --accent-color: #4CAF50; /* A vibrant green for dark mode */
        --link-color: #60AFFF;
    }

    /* General Streamlit component enhancements if needed */
    .stButton>button {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
