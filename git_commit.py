#!/usr/bin/env python3
"""
Git Commit and Push Script
Automates the git add, commit, and push process with user input for commit message.
Configured to push to XIIITrading/Alpha repository.
"""

import subprocess
import sys
import os

# GitHub repository configuration
GITHUB_REPO_URL = "https://github.com/XIIITrading/Alpha.git"
REMOTE_NAME = "origin"

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def get_command_output(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def setup_remote():
    """Ensure the remote repository is properly configured."""
    print(f"🔧 Checking remote repository configuration...")
    
    # Check if remote exists
    remote_url = get_command_output(f"git remote get-url {REMOTE_NAME}")
    
    if remote_url == GITHUB_REPO_URL:
        print(f"✅ Remote '{REMOTE_NAME}' is correctly configured")
        return True
    elif remote_url:
        print(f"⚠️  Remote '{REMOTE_NAME}' exists but points to: {remote_url}")
        print(f"   Expected: {GITHUB_REPO_URL}")
        update = input(f"🤔 Update remote to point to {GITHUB_REPO_URL}? (y/N): ").strip().lower()
        if update in ['y', 'yes']:
            if run_command(f"git remote set-url {REMOTE_NAME} {GITHUB_REPO_URL}", f"Updating remote URL"):
                return True
            else:
                return False
        else:
            print("❌ Remote URL not updated. Please configure manually.")
            return False
    else:
        print(f"📡 Remote '{REMOTE_NAME}' not found. Adding it...")
        if run_command(f"git remote add {REMOTE_NAME} {GITHUB_REPO_URL}", f"Adding remote '{REMOTE_NAME}'"):
            return True
        else:
            return False

def main():
    print("🚀 Git Commit and Push Script")
    print("=" * 40)
    print(f"🎯 Target repository: {GITHUB_REPO_URL}")
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Error: Not in a git repository. Please run this script from your project root.")
        sys.exit(1)
    
    # Setup remote repository
    if not setup_remote():
        print("❌ Failed to configure remote repository.")
        sys.exit(1)
    
    # Get commit message from user
    print("\n💬 Enter your commit message:")
    commit_message = input("> ").strip()
    
    if not commit_message:
        print("❌ Error: Commit message cannot be empty.")
        sys.exit(1)
    
    print(f"\n📝 Commit message: '{commit_message}'")
    
    # Confirm with user
    confirm = input("\n🤔 Proceed with commit and push? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Operation cancelled.")
        sys.exit(0)
    
    print("\n" + "=" * 40)
    
    # Execute git commands
    commands = [
        ("git add .", "Adding all files to staging"),
        (f'git commit -m "{commit_message}"', "Creating commit"),
        (f"git push {REMOTE_NAME}", f"Pushing to {GITHUB_REPO_URL}")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"\n❌ Failed at: {description}")
            print("💡 You may need to check your git status or remote configuration.")
            sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 Success! All changes have been committed and pushed to GitHub.")
    print(f"📋 Commit message: '{commit_message}'")
    print(f"🌐 Repository: {GITHUB_REPO_URL}")

if __name__ == "__main__":
    main() 