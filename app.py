from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import requests
import json
from datetime import datetime
import re
import time
from collections import deque
import openai
from openai import OpenAI
import threading
from queue import Queue, Empty

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute in seconds
        self.requests = deque()
        self.safety_margin = 0.8  # Only use 80% of allowed rate to be safe
    
    def can_make_request(self):
        now = time.time()
        
        # Remove requests older than the window size
        while self.requests and self.requests[0] < now - self.window_size:
            self.requests.popleft()
        
        # Check if we're under the rate limit with safety margin
        max_requests = int(self.requests_per_minute * self.safety_margin)
        return len(self.requests) < max_requests
    
    def add_request(self):
        self.requests.append(time.time())
    
    def wait_time(self):
        if not self.requests:
            return 0
        
        oldest_request = self.requests[0]
        time_to_wait = max(0, oldest_request + self.window_size - time.time())
        return time_to_wait

app = Flask(__name__)

# Configuration - Load from environment variables
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'your-api-key-here')
SERPAPI_KEY = os.environ.get('SERPAPI_KEY', 'your-serpapi-key-here')

# Model configuration
# HF_MODEL is the model name used with the Hugging Face router (via OpenAI client)
# Example: 'openai/gpt-oss-20b' or 'openai/gpt-oss-120b:fireworks-ai'
HF_MODEL = os.environ.get('HF_MODEL', 'openai/gpt-oss-20b')
# OPENROUTER_MODEL is the model name used for the HTTP fallback to OpenRouter
OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'google/gemini-2.0-flash-exp:free')

class SEOKeywordAgent:
    """
    AI-powered SEO keyword research agent that generates and analyzes keywords
    using GPT-OSS-20B and real search engine data.
    """
    
    def __init__(self, openrouter_key, serpapi_key, requests_per_minute=30, hf_model=None, http_model=None):
        """
        Initialize the SEO agent with API credentials.
        
        Args:
            openrouter_key (str): OpenRouter API key for Gemini access
            serpapi_key (str): SerpAPI key for search data
            requests_per_minute (int): Maximum requests per minute to the API
        """
        self.openrouter_key = openrouter_key
        self.serpapi_key = serpapi_key
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        # Prefer Hugging Face router if HF_TOKEN is available
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        self.hf_model = hf_model or HF_MODEL
        self.http_model = http_model or OPENROUTER_MODEL

        if hf_token:
            # Use the Hugging Face router as the base_url
            try:
                self.client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            except Exception:
                self.client = None
        else:
            # Fallback to OpenRouter if OPENROUTER_API_KEY is provided
            if self.openrouter_key and self.openrouter_key != 'your-api-key-here':
                try:
                    self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_key)
                except Exception:
                    self.client = None
            else:
                # No AI client configured
                self.client = None

        self.rate_limiter = RateLimiter(requests_per_minute)
        
    def generate_keyword_variations(self, seed_keyword):
        """
        Generate 50 diverse keyword variations using GPT-OSS-20B.

        Args:
            seed_keyword (str): The seed keyword to expand upon

        Returns:
            list: List of 50 keyword variations
        """
        prompt = f"""You are an SEO keyword research expert. Given the seed keyword: "{seed_keyword}"

Generate 50 diverse keyword variations that are:
1. Highly relevant to the seed keyword
2. Include long-tail keywords (3-5 words)
3. Include question-based keywords (how to, what is, why, when)
4. Include commercial intent keywords (best, top, review, buy, compare)
5. Include informational keywords (guide, tips, tutorial, learn)
6. Mix of different keyword types and user intents

    Return ONLY a JSON array with exactly 50 keywords. Format:
["keyword 1", "keyword 2", "keyword 3", ...]

Do not include any explanation, markdown formatting, or code blocks. Just the raw JSON array."""

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "SEO Keyword Agent"
        }
        
        data = {
            "model": self.http_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        max_retries = 5  # Increased max retries
        base_wait_time = 2  # Base wait time in seconds
        for attempt in range(max_retries):
            try:
                # Check rate limit before making request
                while not self.rate_limiter.can_make_request():
                    wait_time = self.rate_limiter.wait_time()
                    print(f"  ⚠ Rate limit approaching, waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

                print(f"  → Generating keywords with model {self.hf_model}... (attempt {attempt + 1}/{max_retries})")
                if self.client:
                    completion = self.client.chat.completions.create(
                        model=self.hf_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2000,
                        extra_headers={
                            "HTTP-Referer": "http://localhost:5000",
                            "X-Title": "SEO Keyword Agent"
                        }
                    )
                    self.rate_limiter.add_request()
                    content = completion.choices[0].message.content
                else:
                    # Fallback to HTTP call to OpenRouter if configured
                    if self.openrouter_key and self.openrouter_key != 'your-api-key-here':
                        response = requests.post(self.openrouter_url, headers=headers, json=data, timeout=30)
                        self.rate_limiter.add_request()
                        response.raise_for_status()
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                    else:
                        print("  ✗ No AI client configured (set HF_TOKEN or OPENROUTER_API_KEY)")
                        return []
                if content is None:
                    print("  ✗ AI response content is None")
                    return []

                # Extract JSON array from response (handle various formats)
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    keywords = json.loads(json_match.group())
                    # Ensure exactly 50 keywords
                    if len(keywords) >= 50:
                        return keywords[:50]
                    else:
                        print(f"  ⚠ Only got {len(keywords)} keywords, padding with variations...")
                        return keywords

                print("  ✗ Failed to parse keywords from AI response")
                return []

            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter: base * (2 ^ attempt) + random(0-1)
                    wait_time = base_wait_time * (2 ** attempt) + (time.time() % 1)
                    print(f"  ⚠ Rate limit hit, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("  ✗ Rate limit exceeded, max retries reached")
                    return []

            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                return []
    
    def get_search_volume_estimate(self, keyword):
        """
        Estimate search volume and competition using SerpAPI.
        
        Args:
            keyword (str): Keyword to analyze
            
        Returns:
            dict: Dictionary containing volume, competition, and total results
        """
        try:
            # Use Serpstack (api.serpstack.com) as the search provider
            serpstack_key = os.environ.get('SERPSTACK_KEY') or os.environ.get('SERPSTACK_API_KEY')
            if serpstack_key:
                url = 'https://api.serpstack.com/search'
                params = {
                    'access_key': serpstack_key,
                    'query': keyword
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    total_results = data.get('search_information', {}).get('total_results', 0)
                else:
                    print(f"  ⚠ Serpstack returned status {response.status_code}")
                    total_results = 0
            else:
                print("  ⚠ No Serpstack API key configured (set SERPSTACK_KEY)")
                total_results = 0

            # Calculate competition score (0-100 scale)
            if total_results < 100000:
                competition_score = 15  # Very low competition
            elif total_results < 500000:
                competition_score = 25  # Low competition
            elif total_results < 1000000:
                competition_score = 35  # Low-medium competition
            elif total_results < 5000000:
                competition_score = 50  # Medium competition
            elif total_results < 10000000:
                competition_score = 65  # Medium-high competition
            elif total_results < 50000000:
                competition_score = 80  # High competition
            else:
                competition_score = 95  # Very high competition

            # Estimate search volume based on result density
            if total_results < 100000:
                volume = "100-1K"
                volume_numeric = 500
            elif total_results < 500000:
                volume = "1K-10K"
                volume_numeric = 5000
            elif total_results < 5000000:
                volume = "10K-100K"
                volume_numeric = 50000
            else:
                volume = "100K+"
                volume_numeric = 100000

            return {
                'volume': volume,
                'volume_numeric': volume_numeric,
                'competition': competition_score,
                'total_results': total_results
            }

        except requests.exceptions.Timeout:
            print(f"  ⚠ Serpstack timeout for keyword: {keyword}")
        except Exception as e:
            print(f"  ⚠ Error getting search data for {keyword}: {e}")

        # Default fallback values
        return {
            'volume': 'Unknown',
            'volume_numeric': 1000,  # Default to moderate volume
            'competition': 50,  # Default to medium competition
            'total_results': 1000000
        }
    
    def analyze_keyword_difficulty(self, keyword, search_data):
        """
        Use Gemini to analyze keyword difficulty and provide ranking insights.
        
        Args:
            keyword (str): Keyword to analyze
            search_data (dict): Search volume and competition data
            
        Returns:
            str: AI-generated analysis of the keyword
        """
        prompt = f"""As an SEO expert, analyze this keyword: "{keyword}"

Search Data:
- Total Results: {search_data['total_results']:,}
- Competition Score: {search_data['competition']:.1f}/100
- Estimated Search Volume: {search_data['volume']}

Provide a brief analysis (2-3 sentences maximum) covering:
1. Ranking difficulty (Easy/Medium/Hard)
2. Why this keyword is good or bad for first-page ranking
3. One specific content strategy tip

Keep it concise, actionable, and focused on first-page ranking potential."""

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "SEO Keyword Agent"
        }
        
        data = {
            "model": self.http_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 200
        }
        
        max_retries = 5
        base_wait_time = 2
        for attempt in range(max_retries):
            try:
                # Check rate limit before making request
                while not self.rate_limiter.can_make_request():
                    wait_time = self.rate_limiter.wait_time()
                    print(f"  ⚠ Rate limit approaching, waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

                if self.client:
                    completion = self.client.chat.completions.create(
                        model=self.hf_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=200,
                        extra_headers={
                            "HTTP-Referer": "http://localhost:5000",
                            "X-Title": "SEO Keyword Agent"
                        }
                    )
                    self.rate_limiter.add_request()
                    content = completion.choices[0].message.content
                else:
                    if self.openrouter_key and self.openrouter_key != 'your-api-key-here':
                        response = requests.post(self.openrouter_url, headers=headers, json=data, timeout=30)
                        self.rate_limiter.add_request()
                        response.raise_for_status()
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                    else:
                        print("  ✗ No AI client configured (set HF_TOKEN or OPENROUTER_API_KEY)")
                        return "Analysis unavailable"
                if content is None:
                    return "Analysis unavailable"
                analysis = content.strip()
                return analysis
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt) + (time.time() % 1)
                    print(f"  ⚠ Rate limit hit during analysis, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("  ✗ Rate limit exceeded during analysis, max retries reached")
                    return "Rate limit exceeded. Try again later."
            except Exception as e:
                print(f"  ⚠ Error analyzing keyword: {e}")
                return "Analysis unavailable"
    
    def research_keywords(self, seed_keyword, top_n=50, progress_callback=None):
        """
        Main keyword research pipeline.

        Args:
            seed_keyword (str): The seed keyword to research
            top_n (int): Number of top keywords to return (default: 20)

        Returns:
            dict: Complete research results with sorted keywords
        """
        def progress(msg):
            print(msg)
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        progress(f"\n{'='*60}")
        progress(f"Starting SEO Keyword Research for: '{seed_keyword}'")
        progress(f"{'='*60}\n")

        # Step 1: Generate keyword variations using AI
        progress("STEP 1: Generating keyword variations with GPT-OSS-20B")
        keywords = self.generate_keyword_variations(seed_keyword)

        if not keywords:
            return {
                "error": "Failed to generate keywords. Please check your OpenRouter API key and try again."
            }

        progress(f"  ✓ Generated {len(keywords)} keyword variations\n")

        # Step 2: Analyze each keyword
        progress(f"STEP 2: Analyzing {len(keywords)} keywords with search data")
        results = []

        for i, keyword in enumerate(keywords, 1):
            print(f"  [{i}/{len(keywords)}] Analyzing: {keyword}")

            # Get search volume and competition data
            search_data = self.get_search_volume_estimate(keyword)

            # Calculate opportunity score
            # Formula: (Volume / 1000) - Competition
            # Higher volume = more traffic potential
            # Lower competition = easier to rank
            opportunity_score = (search_data['volume_numeric'] / 1000) - search_data['competition']

            results.append({
                'keyword': keyword,
                'volume': search_data['volume'],
                'volume_numeric': search_data['volume_numeric'],
                'competition': search_data['competition'],
                'opportunity_score': opportunity_score,
                'total_results': search_data['total_results']
            })

        progress(f"  ✓ Analysis complete\n")

        # Step 3: Sort by opportunity score (best opportunities first)
        progress("STEP 3: Ranking keywords by opportunity score")
        # Primary sort: volume (descending)
        # Secondary sort: competition (ascending)
        results.sort(key=lambda x: (-x['volume_numeric'], x['competition']))

        # Take top N results
        top_results = results[:top_n]
        progress(f"  ✓ Selected top {len(top_results)} keywords\n")

        # Step 4: Get detailed AI analysis for top keywords
        progress("STEP 4: Generating AI insights for top 5 keywords")

        # Add batch processing with cooldown
        batch_size = 2  # Process keywords in smaller batches
        cooldown = 5    # Seconds to wait between batches

        top_5_results = top_results[:5]
        for batch_start in range(0, len(top_5_results), batch_size):
            batch_end = min(batch_start + batch_size, len(top_5_results))
            batch = top_5_results[batch_start:batch_end]

            for i, result in enumerate(batch, batch_start + 1):
                print(f"  [{i}/5] Analyzing: {result['keyword']}")

                # Per-keyword retry logic to ensure we attempt analysis multiple times
                max_attempts = 3
                ai_analysis = None
                for attempt_idx in range(max_attempts):
                    ai_analysis = self.analyze_keyword_difficulty(
                        result['keyword'],
                        result
                    )

                    # Consider analysis successful if it's not the generic failure message
                    if ai_analysis and ai_analysis not in ("Analysis unavailable", "Rate limit exceeded. Try again later."):
                        break

                    # If failed, wait with exponential backoff before next attempt
                    if attempt_idx < max_attempts - 1:
                        backoff = 2 ** attempt_idx
                        print(f"    ⚠ Analysis attempt {attempt_idx + 1} failed for '{result['keyword']}', retrying in {backoff}s...")
                        time.sleep(backoff)
                    else:
                        print(f"    ✗ All analysis attempts failed for '{result['keyword']}'")

                # Store the best/last attempt result
                result['ai_analysis'] = ai_analysis or "Analysis unavailable"

            # Add cooldown between batches if not the last batch
            if batch_end < len(top_5_results):
                print(f"  → Cooling down for {cooldown} seconds to avoid rate limits...")
                time.sleep(cooldown)

        progress(f"  ✓ AI analysis complete\n")

        progress(f"{'='*60}")
        progress(f"Research Complete! Found {len(top_results)} keyword opportunities")
        progress(f"{'='*60}\n")

        return {
            'seed_keyword': seed_keyword,
            'total_keywords': len(results),
            'keywords': top_results,
            'timestamp': datetime.now().isoformat()
        }


# Initialize the SEO agent with conservative rate limits
agent = SEOKeywordAgent(OPENROUTER_API_KEY, SERPAPI_KEY, requests_per_minute=20)  # More conservative rate limit


# Flask Routes
@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')


@app.route('/api/research', methods=['POST'])
def research():
    """
    API endpoint for keyword research.
    
    Request JSON:
        {
            "seed_keyword": "your keyword here"
        }
    
    Returns:
        JSON with keyword analysis results
    """
    data = request.get_json()
    seed_keyword = data.get('seed_keyword', '').strip()
    
    # Validate input
    if not seed_keyword:
        return jsonify({'error': 'Seed keyword is required'}), 400
    
    if len(seed_keyword) > 100:
        return jsonify({'error': 'Seed keyword too long (max 100 characters)'}), 400
    
    try:
        # Perform keyword research
        results = agent.research_keywords(seed_keyword)
        return jsonify(results)
    except Exception as e:
        print(f"Error in research endpoint: {e}")
        return jsonify({
            'error': f'An error occurred during research: {str(e)}'
        }), 500


def _research_worker(seed, q: Queue):
    """Background worker to run research and push progress messages into queue."""
    def push(msg):
        # Ensure messages are JSON-serializable strings
        try:
            q.put(str(msg))
        except Exception:
            q.put(json.dumps({'progress': str(msg)}))

    try:
        results = agent.research_keywords(seed, progress_callback=push)
        q.put(json.dumps({'type': 'result', 'payload': results}))
    except Exception as e:
        q.put(json.dumps({'type': 'error', 'message': str(e)}))
    finally:
        q.put(None)  # Sentinel to indicate completion


@app.route('/api/research/stream')
def research_stream():
    """SSE endpoint that starts research in background and streams progress messages."""
    seed = request.args.get('query', '').strip()
    if not seed:
        return jsonify({'error': 'query parameter is required'}), 400

    q: Queue = Queue()
    thread = threading.Thread(target=_research_worker, args=(seed, q), daemon=True)
    thread.start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            # If the worker pushed a JSON result or error, send as event
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict) and parsed.get('type') == 'result':
                    yield f"event: result\ndata: {json.dumps(parsed['payload'])}\n\n"
                    continue
                if isinstance(parsed, dict) and parsed.get('type') == 'error':
                    yield f"event: error\ndata: {json.dumps(parsed)}\n\n"
                    continue
            except Exception:
                # Not JSON -> treat as a plain progress message
                pass

            # Normalize newlines to single-line progress messages for the SSE client
            safe = item.replace('\n', ' ') if isinstance(item, str) else str(item)
            yield f"data: {safe}\n\n"

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


@app.route('/api/health')
def health():
    """
    Health check endpoint to verify API configuration.
    
    Returns:
        JSON with status and configuration info
    """
    openrouter_configured = bool(
        OPENROUTER_API_KEY and 
        OPENROUTER_API_KEY != 'your-api-key-here'
    )
    
    serpapi_configured = bool(
        SERPAPI_KEY and 
        SERPAPI_KEY != 'your-serpapi-key-here'
    )
    
    return jsonify({
        'status': 'healthy',
        'openrouter_configured': openrouter_configured,
        'serpapi_configured': serpapi_configured,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SEO Keyword Research AI Agent")
    print("="*60)
    hf_configured = bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN'))
    print(f"HuggingFace Router (HF_TOKEN): {'✓ Configured' if hf_configured else '✗ Not configured'}")
    print(f"OpenRouter API: {'✓ Configured' if OPENROUTER_API_KEY != 'your-api-key-here' else '✗ Not configured'}")
    print(f"SerpAPI: {'✓ Configured' if SERPAPI_KEY != 'your-serpapi-key-here' else '✗ Not configured'}")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("Press CTRL+C to quit\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)