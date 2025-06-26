#!/usr/bin/env python3
"""
Git Commit and Push Script - Enhanced for Multi-Machine Workflow
Automates git add, commit, and push with proper handling for desktop/laptop sync.
Configured to push to XIIITrading/Sirius repository.
"""

import subprocess
import sys
import os
from datetime import datetime

# GitHub repository configuration
GITHUB_REPO_URL = "https://github.com/XIIITrading/Sirius.git"
REMOTE_NAME = "origin"

def run_command(command, description, show_output=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip() and show_output:
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
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def check_for_changes():
    """Check if there are any changes to commit."""
    status = get_command_output("git status --porcelain")
    return bool(status)

def check_unpushed_commits():
    """Check if there are commits that haven't been pushed."""
    try:
        # Fetch latest remote info without merging
        subprocess.run("git fetch", shell=True, capture_output=True)
        # Check for unpushed commits
        unpushed = get_command_output("git log origin/main..HEAD --oneline")
        return unpushed
    except:
        return None

def check_behind_remote():
    """Check if local branch is behind remote."""
    try:
        # Fetch latest remote info without merging
        subprocess.run("git fetch", shell=True, capture_output=True)
        # Check if we're behind
        behind = get_command_output("git log HEAD..origin/main --oneline")
        return behind
    except:
        return None

def get_current_branch():
    """Get the current branch name."""
    return get_command_output("git branch --show-current") or "main"

def push_with_upstream_check():
    """Push to remote, setting upstream if needed."""
    branch = get_current_branch()
    
    # First try a regular push
    result = subprocess.run(f"git push {REMOTE_NAME}", shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Push to {REMOTE_NAME} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    
    # Check if it's the upstream issue
    if "has no upstream branch" in result.stderr or "has no upstream branch" in result.stdout:
        print(f"⚠️  First push from this machine - setting upstream for branch '{branch}'...")
        return run_command(f"git push -u {REMOTE_NAME} {branch}", 
                         f"Setting upstream and pushing to {REMOTE_NAME}/{branch}")
    else:
        # Some other error
        print(f"❌ Push failed: {result.stderr if result.stderr else result.stdout}")
        return False

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
            return run_command(f"git remote set-url {REMOTE_NAME} {GITHUB_REPO_URL}", f"Updating remote URL")
        else:
            print("❌ Remote URL not updated. Please configure manually.")
            return False
    else:
        print(f"📡 Remote '{REMOTE_NAME}' not found. Adding it...")
        return run_command(f"git remote add {REMOTE_NAME} {GITHUB_REPO_URL}", f"Adding remote '{REMOTE_NAME}'")

def sync_check():
    """Check if local is behind remote and offer to pull."""
    behind = check_behind_remote()
    if behind:
        print(f"\n⚠️  Your local branch is behind the remote by these commits:")
        print(behind)
        pull = input("\n🤔 Pull these changes before continuing? (recommended) (Y/n): ").strip().lower()
        if pull != 'n':
            if run_command("git pull", "Pulling latest changes"):
                print("✅ Successfully synchronized with remote!")
                return True
            else:
                print("❌ Pull failed. Please resolve conflicts manually.")
                return False
    return True

def main():
    print("🚀 Git Commit and Push Script")
    print("=" * 40)
    print(f"🎯 Target repository: {GITHUB_REPO_URL}")
    print(f"💻 Machine: {os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'Unknown'))}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("\n❌ Error: Not in a git repository. Please run this script from your project root.")
        sys.exit(1)
    
    # Setup remote repository
    if not setup_remote():
        print("\n❌ Failed to configure remote repository.")
        sys.exit(1)
    
    # Check if we're behind remote (important for multi-machine workflow)
    if not sync_check():
        sys.exit(1)
    
    # Check if there are changes to commit
    if not check_for_changes():
        print("\n✅ Working directory is clean - nothing to commit!")
        
        # Check if we have unpushed commits
        unpushed = check_unpushed_commits()
        if unpushed:
            print(f"\n📤 You have unpushed commits:")
            print(unpushed)
            push_now = input("\n🤔 Push these commits now? (Y/n): ").strip().lower()
            if push_now != 'n':
                if push_with_upstream_check():
                    print("\n🎉 Successfully pushed existing commits!")
                else:
                    print("\n❌ Push failed")
                    sys.exit(1)
        else:
            print("✅ All commits are already pushed to GitHub!")
        sys.exit(0)
    
    # Show what will be committed
    print("\n📝 Changes to be committed:")
    status_output = get_command_output("git status --short")
    if status_output:
        print(status_output)
    
    # Get commit message from user
    print("\n💬 Enter your commit message:")
    commit_message = input("> ").strip()
    
    if not commit_message:
        print("❌ Error: Commit message cannot be empty.")
        sys.exit(1)
    
    print(f"\n📝 Commit message: '{commit_message}'")
    
    # Confirm with user
    confirm = input("\n🤔 Proceed with commit and push? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("❌ Operation cancelled.")
        sys.exit(0)
    
    print("\n" + "=" * 40)
    
    # Add files
    if not run_command("git add .", "Adding all files to staging"):
        print("\n❌ Failed to add files")
        sys.exit(1)
    
    # Commit changes
    commit_result = subprocess.run(f'git commit -m "{commit_message}"', 
                                 shell=True, 
                                 capture_output=True, 
                                 text=True)
    
    if commit_result.returncode != 0:
        if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
            print("ℹ️  No changes to commit after staging")
        else:
            print(f"❌ Commit failed: {commit_result.stderr if commit_result.stderr else commit_result.stdout}")
        sys.exit(1)
    else:
        print("✅ Creating commit completed successfully")
        if commit_result.stdout.strip():
            print(f"Output: {commit_result.stdout.strip()}")
    
    # Push changes
    if not push_with_upstream_check():
        print("\n❌ Failed to push changes")
        print("💡 You may need to pull first or resolve conflicts.")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 Success! All changes have been committed and pushed to GitHub.")
    print(f"📋 Commit message: '{commit_message}'")
    print(f"🌐 Repository: {GITHUB_REPO_URL}")
    print(f"💻 Pushed from: {os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'Unknown'))}")

if __name__ == "__main__":
    main()