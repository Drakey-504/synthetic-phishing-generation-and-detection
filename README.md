### synthetic-phishing-generation-and-detection
Synthetic Phishing Generation and Detection Project for AI Applications in Information Security, Spring 2026

## Setup
 
### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- [Ollama](https://ollama.com) (installed separately as a system application)
 
### Environment Setup
 
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
 
2. Create the conda environment from the lockfile:
   ```bash
   conda env create -f environment.yml
   conda activate phishing
   ```
 
3. Pull the LLM model for phishing email generation:
   ```bash
   ollama pull llama3
   ```