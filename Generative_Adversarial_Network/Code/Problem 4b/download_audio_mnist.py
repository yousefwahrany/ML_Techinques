import os
import subprocess
import sys

def clone_audio_mnist_repo():
    """
    Clone the AudioMNIST repository if it doesn't already exist.
    
    Returns:
    - True if repository was cloned or already exists
    - False if there was an error
    """
    repo_url = "https://github.com/soerenab/AudioMNIST.git"
    repo_name = "AudioMNIST"
    
    # Check if repository already exists
    if os.path.exists(repo_name):
        print(f"Repository {repo_name} already exists in the current directory.")
        return True
    
    try:
        # Attempt to clone the repository
        print(f"Cloning Repository {repo_url}...")
        result = subprocess.run(
            ["git", "clone", repo_url], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        print("Repository successfully cloned.")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print("Error output:", e.stderr)
        return False
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    # Check if git is installed
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Git is not installed. Please install Git to clone the repository.")
        sys.exit(1)
    
    # Clone the repository
    if clone_audio_mnist_repo():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()