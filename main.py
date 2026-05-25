import os
import csv
import json

def load_comments(csv_path):
    """Loads comments from the raw CSV file."""
    comments = []
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return comments
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comments.append(row)
    return comments

def load_prompt(prompt_path):
    """Loads the system prompt instruction."""
    if not os.path.exists(prompt_path):
        print(f"Error: {prompt_path} not found.")
        return ""
    with open(prompt_path, mode='r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'raw_comments.csv')
    prompt_path = os.path.join(base_dir, 'prompt.txt')

    print("--- YouTube Reply Identifier Agent ---")
    print(f"Loading comments from {csv_path}...")
    comments = load_comments(csv_path)
    print(f"Successfully loaded {len(comments)} comments.")

    print(f"Loading prompt from {prompt_path}...")
    prompt = load_prompt(prompt_path)
    if prompt:
        print("Prompt loaded successfully.")
    
    # Placeholder for LLM processing (Day 2/Next Steps)
    print("\nNext steps (Day 2):")
    print("1. Initialize LLM client.")
    print("2. Send raw comments + prompt instructions to LLM.")
    print("3. Retrieve and parse response.")
    print("4. Output final results matching the locked schema:")
    print("   [commenter_name, comment_text, intent_level, intent_reason, recommended_reply, urgency, topic_or_course]")

if __name__ == "__main__":
    main()
