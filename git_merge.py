#!/usr/bin/env python3
"""
Git Pull and Merge Script with Conflict Resolution
Pulls the latest changes from GitHub and merges them, with interactive conflict resolution.
Configured to pull from XIIITrading/Sirius repository.
"""

import subprocess
import sys
import os
import re

# GitHub repository configuration
GITHUB_REPO_URL = "https://github.com/XIIITrading/Sirius.git"
REMOTE_NAME = "origin"

def run_command(command, description, capture_output=True, check=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=capture_output, text=True)
        if check:
            print(f"✅ {description} completed successfully")
        if capture_output and result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ Error during {description}: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
        return e

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
            if run_command(f"git remote set-url {REMOTE_NAME} {GITHUB_REPO_URL}", f"Updating remote URL").returncode == 0:
                return True
            else:
                return False
        else:
            print("❌ Remote URL not updated. Please configure manually.")
            return False
    else:
        print(f"📡 Remote '{REMOTE_NAME}' not found. Adding it...")
        if run_command(f"git remote add {REMOTE_NAME} {GITHUB_REPO_URL}", f"Adding remote '{REMOTE_NAME}'").returncode == 0:
            return True
        else:
            return False

def get_current_branch():
    """Get the current branch name."""
    return get_command_output("git branch --show-current")

def check_git_status():
    """Check if there are any uncommitted changes."""
    return get_command_output("git status --porcelain")

def get_conflicted_files():
    """Get list of files with merge conflicts."""
    result = get_command_output("git diff --name-only --diff-filter=U")
    if result:
        return result.split('\n')
    return []

def show_conflict_diff(file_path):
    """Show the conflict details for a specific file."""
    print(f"\n📄 Conflict details for: {file_path}")
    print("-" * 50)
    
    # Show the conflicting sections
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find conflict markers
        conflicts = re.findall(r'<<<<<<< HEAD(.*?)=======(.*?)>>>>>>> .*', content, re.DOTALL)
        
        if conflicts:
            for i, (local, remote) in enumerate(conflicts, 1):
                print(f"\n🔸 Conflict #{i}:")
                print("📍 Your version (HEAD):")
                print(local.strip()[:200] + "..." if len(local.strip()) > 200 else local.strip())
                print("\n📡 Remote version:")
                print(remote.strip()[:200] + "..." if len(remote.strip()) > 200 else remote.strip())
                print("-" * 30)
    except Exception as e:
        print(f"Error reading file: {e}")

def resolve_conflict(file_path, use_remote=True):
    """Resolve a conflict by choosing either local or remote version."""
    if use_remote:
        # Use remote version
        result = run_command(f"git checkout --theirs {file_path}", f"Using remote version for {file_path}")
    else:
        # Use local version
        result = run_command(f"git checkout --ours {file_path}", f"Using local version for {file_path}")
    
    # Add the resolved file
    run_command(f"git add {file_path}", f"Staging resolved file {file_path}")
    return result.returncode == 0

def handle_merge_conflicts():
    """Interactive conflict resolution."""
    conflicted_files = get_conflicted_files()
    
    if not conflicted_files:
        return True
    
    print(f"\n⚠️  Merge conflicts detected in {len(conflicted_files)} file(s):")
    for i, file in enumerate(conflicted_files, 1):
        print(f"   {i}. {file}")
    
    print("\n🎯 Conflict Resolution Options:")
    print("   1. Overwrite ALL with remote version (quick resolution)")
    print("   2. Keep ALL local versions")
    print("   3. Resolve each file individually")
    print("   4. Abort merge")
    
    choice = input("\n🤔 Choose an option (1-4): ").strip()
    
    if choice == '1':
        # Overwrite all with remote
        print("\n📡 Overwriting all conflicts with remote version...")
        for file in conflicted_files:
            resolve_conflict(file, use_remote=True)
        return True
    
    elif choice == '2':
        # Keep all local
        print("\n📍 Keeping all local versions...")
        for file in conflicted_files:
            resolve_conflict(file, use_remote=False)
        return True
    
    elif choice == '3':
        # Individual resolution
        print("\n🔍 Resolving conflicts individually...")
        for file in conflicted_files:
            show_conflict_diff(file)
            
            while True:
                file_choice = input(f"\n🤔 For '{file}': Use [R]emote, [L]ocal, or [S]kip? ").strip().lower()
                
                if file_choice in ['r', 'remote']:
                    resolve_conflict(file, use_remote=True)
                    break
                elif file_choice in ['l', 'local']:
                    resolve_conflict(file, use_remote=False)
                    break
                elif file_choice in ['s', 'skip']:
                    print(f"⏭️  Skipping {file} - you'll need to resolve manually")
                    break
                else:
                    print("❌ Invalid choice. Please enter R, L, or S.")
        
        # Check if all conflicts were resolved
        remaining_conflicts = get_conflicted_files()
        if remaining_conflicts:
            print(f"\n⚠️  {len(remaining_conflicts)} file(s) still have conflicts:")
            for file in remaining_conflicts:
                print(f"   - {file}")
            print("\n💡 You'll need to resolve these manually before committing.")
            return False
        return True
    
    elif choice == '4':
        # Abort merge
        print("\n🛑 Aborting merge...")
        run_command("git merge --abort", "Aborting merge")
        return False
    
    else:
        print("❌ Invalid choice. Aborting merge...")
        run_command("git merge --abort", "Aborting merge")
        return False

def main():
    print("🚀 Git Pull and Merge Script with Conflict Resolution")
    print("=" * 50)
    print(f"🎯 Target repository: {GITHUB_REPO_URL}")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Error: Not in a git repository. Please run this script from your project root.")
        sys.exit(1)
    
    # Setup remote repository
    if not setup_remote():
        print("❌ Failed to configure remote repository.")
        sys.exit(1)
    
    # Get current branch
    current_branch = get_current_branch()
    if not current_branch:
        print("❌ Error: Could not determine current branch.")
        sys.exit(1)
    
    print(f"📍 Current branch: {current_branch}")
    
    # Check for uncommitted changes
    uncommitted_changes = check_git_status()
    if uncommitted_changes:
        print("\n⚠️  Uncommitted changes detected:")
        print(uncommitted_changes)
        print("\n💡 These changes may cause conflicts during merge.")
        
        stash_choice = input("\n🤔 Stash changes before pulling? (y/N): ").strip().lower()
        if stash_choice in ['y', 'yes']:
            run_command("git stash", "Stashing local changes")
            print("💾 Changes stashed. You can restore them later with 'git stash pop'")
    
    # Show current commit info
    print("\n📋 Current commit information:")
    run_command("git log --oneline -1", "Getting current commit info", capture_output=False)
    
    # Fetch latest changes
    print("\n" + "=" * 50)
    if run_command(f"git fetch {REMOTE_NAME}", f"Fetching latest changes from {REMOTE_NAME}").returncode != 0:
        print("❌ Failed to fetch from remote.")
        sys.exit(1)
    
    # Check if there are new changes
    local_commit = get_command_output(f"git rev-parse {current_branch}")
    remote_commit = get_command_output(f"git rev-parse {REMOTE_NAME}/{current_branch}")
    
    if local_commit == remote_commit:
        print("\n✅ Already up to date!")
        sys.exit(0)
    
    # Show what will be merged
    print("\n📊 Changes to be merged:")
    run_command(f"git log --oneline {current_branch}..{REMOTE_NAME}/{current_branch}", 
                "Showing incoming changes", capture_output=False)
    
    # Confirm merge
    confirm = input("\n🤔 Proceed with merge? (yes/NO): ").strip().lower()
    if confirm != 'yes':
        print("❌ Merge cancelled.")
        sys.exit(0)
    
    # Perform pull (fetch + merge)
    print("\n" + "=" * 50)
    pull_result = run_command(f"git pull {REMOTE_NAME} {current_branch}", 
                             f"Pulling and merging from {REMOTE_NAME}/{current_branch}", 
                             check=False)
    
    # Check if merge was successful or if there are conflicts
    if pull_result.returncode == 0:
        print("\n🎉 Success! Pull and merge completed without conflicts.")
    else:
        print("\n⚠️  Merge conflicts detected!")
        if handle_merge_conflicts():
            # Check if we need to complete the merge
            if get_conflicted_files():
                print("\n❌ Some conflicts remain unresolved. Please resolve manually.")
                sys.exit(1)
            else:
                # Complete the merge
                print("\n🔄 Completing merge...")
                commit_result = run_command("git commit --no-edit", "Completing merge commit", check=False)
                if commit_result.returncode == 0:
                    print("\n🎉 Merge completed successfully!")
                else:
                    print("\n💡 Merge already completed or no changes to commit.")
        else:
            print("\n❌ Merge resolution failed or was aborted.")
            sys.exit(1)
    
    # Show final status
    print("\n📋 Updated commit information:")
    run_command("git log --oneline -1", "Getting updated commit info", capture_output=False)
    print("\n📊 Final repository status:")
    run_command("git status", "Checking final status", capture_output=False)

if __name__ == "__main__":
    main()