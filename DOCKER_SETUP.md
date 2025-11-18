# YouTube Shorts Generator - Docker & Docker Compose Setup

This guide provides instructions for setting up and running the application using Docker and Docker Compose. This is the recommended method for both development and production as it simplifies dependency management and ensures a consistent environment.

## Prerequisites

1.  **Git:** To clone the repository.
2.  **Docker:** [Install Docker](https://docs.docker.com/engine/install/) for your operating system.
3.  **Docker Compose:** Usually included with Docker Desktop. If not, [install Docker Compose](https://docs.docker.com/compose/install/).
4.  **NVIDIA GPU:** The host machine must have an NVIDIA GPU.
5.  **NVIDIA Container Toolkit:** To enable GPU support in Docker containers. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Quickstart

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/evgeniy-kachan/youtube-shorts-generator.git
    cd youtube-shorts-generator
    ```

2.  **Create an environment file:**
    Copy the example environment file. You can customize it later if needed.
    ```bash
    cp .env.example .env
    ```

3.  **Build and run the application:**
    This command will build the Docker images for the frontend and backend, pull the Ollama image, and start all services in the background.
    ```bash
    docker-compose up --build -d
    ```

4.  **Pull the Ollama model:**
    Once the containers are running, you need to pull the LLM model that the application will use.
    ```bash
    docker-compose exec ollama ollama pull llama3.1:8b
    ```
    *(You can replace `llama3.1:8b` with any other model specified in your `.env` file).*

5.  **Access the application:**
    Open your web browser and navigate to `http://localhost`. The React frontend should be visible and ready to accept YouTube links or local file uploads.

## Usage

*   **API:** The backend API is available at `http://localhost:8000/docs`.
*   **Frontend:** The user interface is at `http://localhost`.
*   **Ollama:** The Ollama API is at `http://localhost:11434`.

## Managing the Application

*   **View logs:**
    ```bash
    # View logs for all services
    docker-compose logs -f

    # View logs for a specific service (e.g., backend)
    docker-compose logs -f backend
    ```

*   **Stop the application:**
    This command stops and removes the containers.
    ```bash
    docker-compose down
    ```

*   **Stop and remove volumes (clears all data):**
    Use this command if you want to start fresh, including deleting the Ollama models and model caches.
    ```bash
    docker-compose down -v
    ```

*   **Updating the application:**
    To update to the latest version from the Git repository:
    ```bash
    # 1. Stop the current services
    docker-compose down

    # 2. Pull the latest code
    git pull origin main

    # 3. Rebuild and restart the services
    docker-compose up --build -d
    ```

## Notes on GPU Access

The `docker-compose.yml` file is configured to grant the `backend` and `ollama` services access to the host's NVIDIA GPU. This requires the NVIDIA Container Toolkit to be properly installed and configured on the host machine. If you encounter GPU-related errors, ensure that `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works correctly on your host.
