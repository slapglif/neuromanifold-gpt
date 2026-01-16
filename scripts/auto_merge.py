import json
import subprocess
import sys
import os

def run(cmd, cwd=None):
    print(f"Running: {cmd} (cwd={cwd or '.'})")
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None

def get_prs():
    json_out = run("gh pr list --limit 100 --json number,title,headRefName,mergeable,url")
    if not json_out:
        return []
    return json.loads(json_out)

def get_worktrees():
    output = run("git worktree list --porcelain")
    if not output:
        return {}
    
    worktrees = {}
    current_worktree = None
    for line in output.splitlines():
        if line.startswith("worktree "):
            current_worktree = line.split(" ", 1)[1]
        elif line.startswith("branch ") and current_worktree:
            branch_ref = line.split(" ", 1)[1]
            if branch_ref.startswith("refs/heads/"):
                branch = branch_ref.replace("refs/heads/", "")
                worktrees[branch] = current_worktree
            current_worktree = None
    return worktrees

def attempt_simple_merge(pr):
    print(f"Attempting to merge PR #{pr['number']}: {pr['title']}")
    # Try merging via GH
    res = run(f"gh pr merge {pr['number']} --merge --delete-branch")
    if res is not None:
        print(f"Successfully merged PR #{pr['number']}")
        return True
    return False

def resolve_conflicts_and_update(pr, worktree_path):
    print(f"Attempting to resolve conflicts for PR #{pr['number']} in {worktree_path}")
    branch = pr['headRefName']
    
    # Stash any local changes to ensure clean merge state
    run("git stash", cwd=worktree_path)
    
    # Ensure we are on the right branch and pulled
    run(f"git checkout {branch}", cwd=worktree_path)
    run(f"git pull origin {branch}", cwd=worktree_path)
    
    # Try merge master
    run("git fetch origin", cwd=worktree_path)
    merge_res = run("git merge origin/master --no-commit --no-ff", cwd=worktree_path)
    
    if merge_res is None: # Conflict or merge failure
        print("Merge failed or conflicts detected.")
        
        # Check if we are actually in a merge state
        git_dir = run("git rev-parse --git-dir", cwd=worktree_path)
        if not os.path.exists(os.path.join(git_dir, "MERGE_HEAD")):
             print("Merge did not start (likely due to local changes or other error). Aborting.")
             return False

        print("Attempting heuristic resolution...")
        
        # Get list of unmerged files
        status = run("git status --porcelain", cwd=worktree_path)
        files = [line.split()[-1] for line in status.splitlines() if line.startswith("UU") or line.startswith("AA")]
        
        resolved_count = 0
        for f in files:
            if f.endswith(".auto-claude-status") or f.endswith(".claude_settings.json") or f == "pytest.ini":
                # Always accept ours for task status/settings
                print(f"Resolving {f} using ours")
                run(f"git checkout --ours {f}", cwd=worktree_path)
                run(f"git add {f}", cwd=worktree_path)
                resolved_count += 1
            elif f == "uv.lock":
                 print(f"Resolving {f} using ours (lockfile)")
                 run(f"git checkout --ours {f}", cwd=worktree_path)
                 run(f"git add {f}", cwd=worktree_path)
                 resolved_count += 1
            else:
                print(f"Unknown conflict in {f}. Skipping auto-resolution.")
        
        # Check if all resolved
        status_after = run("git status --porcelain", cwd=worktree_path)
        if "UU" in status_after or "AA" in status_after:
             print("Could not resolve all conflicts automatically.")
             run("git merge --abort", cwd=worktree_path)
             # Pop stash if we aborted? Maybe safer to leave it stashed to avoid confusion
             return False
        
        # Commit
        run("git commit -m 'Merge master and resolve conflicts'", cwd=worktree_path)
        run(f"git push origin {branch}", cwd=worktree_path)
        
        # Retry merge
        return attempt_simple_merge(pr)
        
    else:
        # Merge successful without conflict?
        print("Merge successful locally. Pushing...")
        run(f"git push origin {branch}", cwd=worktree_path)
        return attempt_simple_merge(pr)

def main():
    prs = get_prs()
    print(f"Found {len(prs)} PRs")
    
    worktrees = get_worktrees()
    print(f"Found {len(worktrees)} worktrees")
    
    for pr in prs:
        if pr['mergeable'] == 'MERGEABLE':
            attempt_simple_merge(pr)
        else:
            branch = pr['headRefName']
            if branch in worktrees:
                print(f"PR #{pr['number']} ({branch}) has a worktree at {worktrees[branch]}. Updating...")
                resolve_conflicts_and_update(pr, worktrees[branch])
            else:
                print(f"PR #{pr['number']} ({branch}) has NO active worktree. Skipping manual update.")

if __name__ == "__main__":
    main()