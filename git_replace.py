#!/usr/bin/env python3
"""
Git Latest Pull and Replace Script
Pulls the latest changes from GitHub and performs a hard reset to remove any local changes.
This effectively replaces the local codebase with the latest remote version.
Configured to pull from XIIITrading/Alpha repository.
"""

import subprocess
import sys
import os

# GitHub repository configuration
GITHUB_REPO_URL = "https://github.com/XIIITrading/Alpha.git"
REMOTE_NAME = "origin"

def run_command(command, description, capture_output=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=capture_output, text=True)
        print(f"✅ {description} completed successfully")
        if capture_output and result.stdout.strip():
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

def get_current_branch():
    """Get the current branch name."""
    try:
        result = subprocess.run("git branch --show-current", shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def get_remote_url():
    """Get the remote URL for the current repository."""
    try:
        result = subprocess.run(f"git remote get-url {REMOTE_NAME}", shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def check_git_status():
    """Check if there are any uncommitted changes."""
    try:
        result = subprocess.run("git status --porcelain", shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def main():
    print("🚀 Git Latest Pull and Replace Script")
    print("=" * 50)
    print(f"🎯 Target repository: {GITHUB_REPO_URL}")
    print("⚠️  WARNING: This will discard ALL local changes!")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Error: Not in a git repository. Please run this script from your project root.")
        sys.exit(1)
    
    # Setup remote repository
    if not setup_remote():
        print("❌ Failed to configure remote repository.")
        sys.exit(1)
    
    # Get current branch and remote info
    current_branch = get_current_branch()
    remote_url = get_remote_url()
    
    if not current_branch:
        print("❌ Error: Could not determine current branch.")
        sys.exit(1)
    
    if not remote_url:
        print("❌ Error: No remote 'origin' found. Please configure your remote repository.")
        sys.exit(1)
    
    print(f"📍 Current branch: {current_branch}")
    print(f"🌐 Remote URL: {remote_url}")
    
    # Check for uncommitted changes
    uncommitted_changes = check_git_status()
    if uncommitted_changes:
        print("\n⚠️  Uncommitted changes detected:")
        print(uncommitted_changes)
        print("\n💡 These changes will be permanently lost!")
    else:
        print("\n✅ No uncommitted changes detected.")
    
    # Show current commit info
    print("\n📋 Current commit information:")
    run_command("git log --oneline -1", "Getting current commit info", capture_output=False)
    
    # Confirm with user
    print("\n" + "=" * 50)
    confirm = input("🤔 Are you sure you want to pull latest and reset? This will discard ALL local changes! (yes/NO): ").strip().lower()
    if confirm != 'yes':
        print("❌ Operation cancelled.")
        sys.exit(0)
    
    print("\n" + "=" * 50)
    
    # Execute git commands
    commands = [
        (f"git fetch {REMOTE_NAME}", f"Fetching latest changes from {GITHUB_REPO_URL}"),
        (f"git reset --hard {REMOTE_NAME}/{current_branch}", f"Hard resetting to {REMOTE_NAME}/{current_branch}"),
        ("git clean -fd", "Removing untracked files and directories")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"\n❌ Failed at: {description}")
            print("💡 You may need to check your git status or remote configuration.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Success! Your local repository has been updated to match the latest remote version.")
    print(f"🌐 Repository: {GITHUB_REPO_URL}")
    print("📋 Updated commit information:")
    run_command("git log --oneline -1", "Getting updated commit info", capture_output=False)
    
    # Show final status
    print("\n📊 Final repository status:")
    run_command("git status", "Checking final status", capture_output=False)

if __name__ == "__main__":
    main()