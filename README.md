# RAG-based Document Query System

## Project Structure

```
/
├── backend/                 # Backend service
│   ├── src/                # Source code
│   ├── tests/              # Test files
│   ├── config/             # Configuration
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile         
│   ├── .env.sample        # Sample env for local development
│   └── .env.docker.sample # Sample env for Docker
├── frontend/               # Frontend service
│   ├── src/               # React source code
│   ├── public/            # Static files
│   ├── package.json       # Node dependencies
│   ├── .env.sample        # Sample env for local development
│   └── .env.docker.sample # Sample env for Docker
├── docker-compose.yml     # Docker services configuration
└── README.md             # Project documentation
```

## Features

[Previous Features section remains the same...]

## Environment Setup

The system uses different environment configurations for local development and Docker:

### Backend Environment
1. Local Development (backend/.env):
   - Copy backend/.env.sample to backend/.env
   - Configure with local paths and settings
   - Required variables:
     ```bash
     OPENAI_API_KEY=your_openai_api_key
     CHROMA_PERSIST_DIR=./chroma_db
     CHROMA_COLLECTION_NAME=your_collection_name
     ```

2. Docker Environment (backend/.env.docker):
   - Copy backend/.env.docker.sample to backend/.env.docker
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
2. Set up backend:
   ```bash
   cd backend
   ./setup_data_dirs.sh
   pip install -r requirements.txt
   cp .env.sample .env
   # Configure .env with your settings
   ```
3. Set up frontend:
   ```bash
   cd frontend
   npm install
   cp .env.sample .env
   # Configure .env with your settings
   ```
4. Run backend:
   ```bash
   cd backend
   python src/app.py
   ```
5. Run frontend:
   ```bash
   cd frontend
   npm start
   ```

### Docker Deployment

1. Clone the repository
2. Set up environment files:
   ```bash
   # Backend
   cp backend/.env.docker.sample backend/.env.docker
   # Frontend
   cp frontend/.env.docker.sample frontend/.env.docker
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

## Data Storage

### Local Development
- Data is stored in backend/data/
- Cache directories are created by setup_data_dirs.sh
- Manual cleanup required if needed

### Docker Environment
- Data is stored in Docker named volumes
- Volumes are automatically managed by Docker
- Use `docker compose down -v` to clean up all data
- Uploads directory is mounted from backend/uploads/

## Development

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Formatting
```bash
# Backend
cd backend
black .

# Frontend
cd frontend
npm run format
```

[Previous API Endpoints section remains the same...]
