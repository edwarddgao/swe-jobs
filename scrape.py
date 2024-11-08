from jobspy import scrape_jobs
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import csv
import time
import subprocess
from pathlib import Path

def load_existing_jobs(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def calculate_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2):
        return 0
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([str(text1), str(text2)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0

def deduplicate_jobs(new_jobs_df, existing_jobs_df, similarity_threshold=0.85):
    if existing_jobs_df.empty:
        return new_jobs_df
    
    # First deduplicate by ID
    unique_jobs = new_jobs_df[~new_jobs_df['id'].isin(existing_jobs_df['id'])]
    
    # Then check content similarity for remaining jobs
    duplicates = []
    for _, new_job in unique_jobs.iterrows():
        for _, existing_job in existing_jobs_df.iterrows():
            title_sim = calculate_similarity(new_job['title'], existing_job['title'])
            desc_sim = calculate_similarity(new_job['description'], existing_job['description'])
            combined_sim = (title_sim * 0.6) + (desc_sim * 0.4)
            
            if combined_sim > similarity_threshold:
                duplicates.append(new_job['id'])
                break
    
    return unique_jobs[~unique_jobs['id'].isin(duplicates)]

def git_push_changes(repo_path: Path, commit_message: str):
    """Push changes to GitHub repository."""
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        print("Successfully pushed changes to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to GitHub: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def scrape_and_save():
    # Set up paths
    base_dir = Path(__file__).parent
    csv_path = base_dir / "data" / "jobs_database.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs(csv_path.parent, exist_ok=True)
    
    # Load existing jobs
    existing_jobs = load_existing_jobs(csv_path)
    
    # Scrape jobs for each job type separately
    all_new_jobs = []
    
    try:
        # Full-time jobs
        full_time_jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
            search_term='"entry level software engineer" OR "new grad software engineer"',
            job_type="fulltime",
            results_wanted=50,
            hours_old=24,
            description_format="markdown"
        )
        all_new_jobs.append(full_time_jobs)
        
        # Add delay between scrapes to avoid rate limiting
        time.sleep(30)
        
        # Internship jobs
        internship_jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
            search_term='"software engineering intern" OR "software developer intern"',
            job_type="internship",
            results_wanted=50,
            hours_old=24,
            description_format="markdown"
        )
        all_new_jobs.append(internship_jobs)
    except Exception as e:
        print(f"Error during scraping: {e}")
        return
    
    # Combine all jobs
    new_jobs = pd.concat(all_new_jobs, ignore_index=True)
    
    # Filter for entry-level positions
    entry_level_keywords = [
        'entry level', 'entry-level', 'junior', 'associate', 
        'intern', 'internship', 'new grad', 'graduate'
    ]
    
    # Filter and create a copy to avoid SettingWithCopyWarning
    filtered_jobs = new_jobs[
        new_jobs['job_level'].str.lower().str.contains('entry|junior', na=False) |
        new_jobs['title'].str.lower().str.contains('|'.join(entry_level_keywords), na=False) |
        new_jobs['description'].str.lower().str.contains('|'.join(entry_level_keywords), na=False)
    ].copy()
    
    # Deduplicate
    unique_jobs = deduplicate_jobs(filtered_jobs, existing_jobs)
    
    # Add timestamp
    unique_jobs.loc[:, 'scrape_timestamp'] = datetime.now().isoformat()
    
    # Combine with existing jobs and save
    if not existing_jobs.empty:
        combined_jobs = pd.concat([existing_jobs, unique_jobs], ignore_index=True)
    else:
        combined_jobs = unique_jobs
        
    # Save to CSV
    combined_jobs.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    # Generate commit message with statistics
    commit_message = f"Update job database: +{len(unique_jobs)} new jobs, total {len(combined_jobs)}"
    
    # Push changes to GitHub
    git_push_changes(base_dir, commit_message)
    
    print(f"Added {len(unique_jobs)} new unique entry-level/intern jobs")
    print(f"Total jobs in database: {len(combined_jobs)}")

if __name__ == "__main__":
    scrape_and_save()