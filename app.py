import os
import json
from openai import OpenAI
from flask import Flask, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables (useful for local dev, ignored in Railway)
load_dotenv()

# Initialize Flask App, OpenAI, and Supabase
app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def analyze_and_store_image(image_url: str):
    """
    Analyzes an image to get a description and tags, creates an embedding,
    and upserts the data to a Supabase vector table.
    """
    try:
        # Step 1: Get structured description and tags from OpenAI
        print(f"Analyzing image from URL: {image_url}")
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image carefully. Provide a detailed, one-paragraph description and a list of exactly 15 relevant tags.
                                       Format your response as a single JSON object with two keys: "description" (string) and "tags" (an array of 15 strings)."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "low"},
                        }
                    ],
                }
            ],
            max_tokens=500
        )
        
        result_json = json.loads(analysis_response.choices[0].message.content)
        description = result_json.get("description", "")
        tags = result_json.get("tags", [])

        if not description or not tags:
            raise ValueError("Failed to get a valid description or tags from the AI model.")
        
        print(f"Description: {description}")

        # Step 2: Create a combined text for embedding
        combined_content = f"Description: {description} Tags: {', '.join(tags)}"

        # Step 3: Generate the vector embedding
        print("Generating embedding...")
        embedding_response = client.embeddings.create(
            input=[combined_content],
            model="text-embedding-3-small"
        )
        embedding = embedding_response.data[0].embedding

        # Step 4: Prepare data and upsert to Supabase
        data_to_upsert = {
            "content": combined_content,
            "embedding": embedding,
            "metadata": {
                "imageUrl": image_url,
                "description": description,
                "tags": tags
            }
        }

        print("Upserting data to Supabase...")
        response = supabase.table("image_vectors").insert(data_to_upsert).execute()
        
        if response.data is None and response.error is not None:
             raise Exception(f"Supabase error: {response.error.message}")

        print("Data successfully stored in Supabase.")
        return {"description": description, "tags": tags}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

# --- This is the ONLY endpoint now ---
@app.route('/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "Missing 'image_url' in request body"}), 400

    image_url = data['image_url']
    result = analyze_and_store_image(image_url)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result), 200

# --- Standard code to run the app ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
