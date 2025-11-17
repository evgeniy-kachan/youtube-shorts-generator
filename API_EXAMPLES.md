# 📡 API Examples

Примеры использования API YouTube Shorts Generator.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### 2. Start Video Analysis

```bash
POST /api/video/analyze
Content-Type: application/json

{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Analysis started"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/video/analyze \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/video/analyze",
    json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
)

task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")
```

### 3. Check Task Status

```bash
GET /api/video/task/{task_id}
```

**Response (Processing):**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "progress": 0.5,
  "message": "Analyzing content...",
  "result": null
}
```

**Response (Completed):**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 1.0,
  "message": "Analysis completed",
  "result": {
    "video_id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "duration": 212.0,
    "segments": [
      {
        "id": "segment_0",
        "start_time": 10.5,
        "end_time": 45.3,
        "duration": 34.8,
        "text_en": "Never gonna give you up, never gonna let you down...",
        "text_ru": "Никогда не брошу тебя, никогда не подведу...",
        "highlight_score": 0.85,
        "criteria_scores": {
          "information_density": 0.7,
          "emotional_intensity": 0.9,
          "hook_potential": 0.8,
          ...
        }
      }
    ]
  }
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/video/task/123e4567-e89b-12d3-a456-426614174000
```

**Python Example with Polling:**
```python
import requests
import time

def wait_for_task(task_id, max_wait=600, poll_interval=5):
    """Wait for task to complete."""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"http://localhost:8000/api/video/task/{task_id}")
        data = response.json()
        
        status = data["status"]
        progress = data["progress"]
        message = data["message"]
        
        print(f"[{progress*100:.0f}%] {message}")
        
        if status == "completed":
            return data["result"]
        elif status == "failed":
            raise Exception(f"Task failed: {message}")
        
        time.sleep(poll_interval)
    
    raise TimeoutError("Task timeout")

# Usage
task_id = "123e4567-e89b-12d3-a456-426614174000"
result = wait_for_task(task_id)
print(f"Found {len(result['segments'])} interesting segments")
```

### 4. Process Selected Segments

```bash
POST /api/video/process
Content-Type: application/json

{
  "video_id": "dQw4w9WgXcQ",
  "segment_ids": ["segment_0", "segment_1", "segment_3"]
}
```

**Response:**
```json
{
  "task_id": "234e5678-e89b-12d3-a456-426614174111",
  "status": "pending",
  "message": "Processing started"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/video/process \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "segment_ids": ["segment_0", "segment_1"]
  }'
```

### 5. Download Processed Segment

```bash
GET /api/video/download/{video_id}/{segment_id}
```

**cURL Example:**
```bash
curl -O http://localhost:8000/api/video/download/dQw4w9WgXcQ/segment_0
```

**Python Example:**
```python
import requests

def download_segment(video_id, segment_id, output_path):
    """Download processed video segment."""
    url = f"http://localhost:8000/api/video/download/{video_id}/{segment_id}"
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded: {output_path}")

# Usage
download_segment("dQw4w9WgXcQ", "segment_0", "clip_1.mp4")
```

### 6. Cleanup

```bash
DELETE /api/video/cleanup/{video_id}
```

**Response:**
```json
{
  "message": "Cleanup completed"
}
```

## Complete Workflow Example

### Python Script

```python
import requests
import time
from pathlib import Path

class YouTubeShortsGenerator:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_video(self, youtube_url):
        """Start video analysis."""
        response = requests.post(
            f"{self.base_url}/api/video/analyze",
            json={"youtube_url": youtube_url}
        )
        return response.json()["task_id"]
    
    def get_task_status(self, task_id):
        """Get task status."""
        response = requests.get(f"{self.base_url}/api/video/task/{task_id}")
        return response.json()
    
    def wait_for_task(self, task_id, max_wait=600):
        """Wait for task completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            data = self.get_task_status(task_id)
            
            if data["status"] == "completed":
                return data["result"]
            elif data["status"] == "failed":
                raise Exception(f"Task failed: {data['message']}")
            
            print(f"Progress: {data['progress']*100:.0f}% - {data['message']}")
            time.sleep(5)
        
        raise TimeoutError("Task timeout")
    
    def process_segments(self, video_id, segment_ids):
        """Process selected segments."""
        response = requests.post(
            f"{self.base_url}/api/video/process",
            json={
                "video_id": video_id,
                "segment_ids": segment_ids
            }
        )
        return response.json()["task_id"]
    
    def download_segment(self, video_id, segment_id, output_dir="./downloads"):
        """Download processed segment."""
        Path(output_dir).mkdir(exist_ok=True)
        
        url = f"{self.base_url}/api/video/download/{video_id}/{segment_id}"
        output_path = Path(output_dir) / f"{video_id}_{segment_id}.mp4"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path

# Usage
generator = YouTubeShortsGenerator()

# 1. Analyze video
print("Starting analysis...")
youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
task_id = generator.analyze_video(youtube_url)
print(f"Task ID: {task_id}")

# 2. Wait for analysis
print("Waiting for analysis to complete...")
result = generator.wait_for_task(task_id)
print(f"\nAnalysis complete!")
print(f"Video: {result['title']}")
print(f"Duration: {result['duration']:.1f}s")
print(f"Found {len(result['segments'])} segments\n")

# 3. Show top segments
segments = sorted(result['segments'], key=lambda x: x['highlight_score'], reverse=True)
print("Top 5 segments:")
for i, seg in enumerate(segments[:5], 1):
    print(f"{i}. Score: {seg['highlight_score']:.2f} | "
          f"{seg['start_time']:.1f}s - {seg['end_time']:.1f}s | "
          f"{seg['text_ru'][:50]}...")

# 4. Select top 3 segments
selected_ids = [seg['id'] for seg in segments[:3]]
print(f"\nProcessing top 3 segments: {selected_ids}")

# 5. Process segments
task_id = generator.process_segments(result['video_id'], selected_ids)
print("Waiting for processing to complete...")
processed_result = generator.wait_for_task(task_id)

# 6. Download all processed segments
print("\nDownloading processed segments...")
for segment in processed_result['processed_segments']:
    output_path = generator.download_segment(
        result['video_id'],
        segment['segment_id']
    )
    print(f"Downloaded: {output_path}")

print("\n✅ All done!")
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Common Errors

**404 - Not Found:**
```json
{
  "detail": "Task not found"
}
```

**400 - Bad Request:**
```json
{
  "detail": "Invalid YouTube URL"
}
```

**500 - Internal Server Error:**
```json
{
  "detail": "Video is too long: 150.5 minutes (max 120 minutes)"
}
```

### Python Error Handling Example

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/api/video/analyze",
        json={"youtube_url": "invalid-url"}
    )
    response.raise_for_status()
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.json()}")
    
except requests.exceptions.ConnectionError:
    print("Cannot connect to server")
    
except Exception as e:
    print(f"Error: {e}")
```

## Rate Limiting

Currently no rate limiting is implemented. For production:

- Implement rate limiting per IP
- Add authentication
- Add queue system for multiple users

## WebSocket Support (Future)

For real-time progress updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/task/123e4567');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress * 100}%`);
};
```

## Batch Processing Example

```python
import asyncio
import aiohttp

async def process_multiple_videos(urls):
    """Process multiple videos concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for url in urls:
            task = analyze_and_process_video(session, url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

async def analyze_and_process_video(session, youtube_url):
    """Analyze and process a single video."""
    base_url = "http://localhost:8000"
    
    # Start analysis
    async with session.post(
        f"{base_url}/api/video/analyze",
        json={"youtube_url": youtube_url}
    ) as resp:
        data = await resp.json()
        task_id = data["task_id"]
    
    # Wait for completion
    while True:
        async with session.get(f"{base_url}/api/video/task/{task_id}") as resp:
            data = await resp.json()
            
            if data["status"] == "completed":
                return data["result"]
            elif data["status"] == "failed":
                raise Exception(data["message"])
            
            await asyncio.sleep(5)

# Usage
urls = [
    "https://www.youtube.com/watch?v=VIDEO1",
    "https://www.youtube.com/watch?v=VIDEO2",
    "https://www.youtube.com/watch?v=VIDEO3",
]

results = asyncio.run(process_multiple_videos(urls))
print(f"Processed {len(results)} videos")
```

## Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with invalid URL
curl -X POST http://localhost:8000/api/video/analyze \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "not-a-url"}'

# Test non-existent task
curl http://localhost:8000/api/video/task/non-existent-id
```

---

For more information, see the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

