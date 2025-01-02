# RAG-based Document Query System

[Previous sections until Environment Setup remain the same...]

## Environment Setup

The system uses different environment configurations for local development and Docker:

### Backend Environment
1. Local Development (.env):
   - Copy .env.sample to .env
   - Configure with local paths and settings
   - Required variables:
     ```bash
     OPENAI_API_KEY=your_openai_api_key
     CHROMA_PERSIST_DIR=./chroma_db
     CHROMA_COLLECTION_NAME=your_collection_name
     ```

2. Docker Environment (.env.docker):
   - Copy .env.docker.sample to .env.docker
   - Uses Docker-specific paths and settings
   - Container paths are automatically mounted as volumes

### Frontend Environment
1. Local Development (frontend/.env):
   - Copy frontend/.env.sample to frontend/.env
   - Configure API URL for local development:
     ```bash
     REACT_APP_API_URL=http://localhost:5000
     ```

2. Docker Environment (frontend/.env.docker):
   - Copy frontend/.env.docker.sample to frontend/.env.docker
   - Uses Docker-specific settings:
     ```bash
     REACT_APP_API_URL=http://localhost:5001
     ```

## Installation

### Local Development

1. Clone the repository
2. Set up data directories:
   ```bash
   ./setup_data_dirs.sh
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment files:
   ```bash
   cp .env.sample .env
   cp frontend/.env.sample frontend/.env
   ```
5. Configure your .env files with appropriate values
6. Run the backend:
   ```bash
   python src/app.py
   ```
7. Install and run frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

### Docker Deployment

1. Clone the repository
2. Set up environment files:
   ```bash
   cp .env.docker.sample .env.docker        # Backend Docker environment
   cp frontend/.env.docker.sample frontend/.env.docker  # Frontend Docker environment
   ```
3. Configure your .env.docker files with appropriate values
4. Start the application:
   ```bash
   docker compose up -d
   ```
5. To stop and clean up:
   ```bash
   docker compose down -v
   ```

[Rest of the file remains the same...]
