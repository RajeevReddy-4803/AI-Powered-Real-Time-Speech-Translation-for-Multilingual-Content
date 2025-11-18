# GitHub Setup Instructions

Follow these steps to add your changes to GitHub.

## Step 1: Review What Will Be Committed

First, check what files have changed:

```powershell
git status
```

You should see:
- Modified files (preprocessing, models, README, etc.)
- New `lingua_kit/` directory (untracked)

## Step 2: Review Changes (Optional but Recommended)

Check what changed in specific files:

```powershell
# See changes in README
git diff README.md

# See changes in a specific file
git diff models/whisper/train_whisper_asr.py
```

## Step 3: Stage Files for Commit

### Option A: Add All Changes (Recommended for first commit)

```powershell
# Add all modified and new files
git add .

# Or add specific files/directories
git add README.md
git add .gitignore
git add preprocess/
git add models/
git add lingua_kit/
git add requirements.txt
```

### Option B: Add Files Selectively

If you want to review each file:

```powershell
# Add modified files
git add README.md
git add .gitignore
git add requirements.txt
git add preprocess/
git add models/

# Add the new lingua_kit directory
git add lingua_kit/
```

## Step 4: Verify What's Staged

Check what will be committed:

```powershell
git status
```

You should see files listed under "Changes to be committed".

**Important:** Make sure you're NOT committing:
- Large audio files (*.wav, *.mp3, etc.)
- Model checkpoints
- Data directories
- Cache files

These should be ignored by `.gitignore`.

## Step 5: Commit Changes

Create a commit with a descriptive message:

```powershell
git commit -m "Refactor: Rename package to lingua_kit and integrate translation modules

- Renamed rajeev_polyglot to lingua_kit throughout codebase
- Integrated QALAMAI translation modules into main project
- Updated preprocessing pipeline with improved path handling
- Enhanced Whisper training script with auto-detection features
- Added end-to-end orchestration script (atlas_driver.py)
- Removed redundant utility scripts and outdated documentation
- Updated all imports, references, and documentation
- Maintained by Karukonda Rajeev Reddy"
```

Or a shorter version:

```powershell
git commit -m "Refactor: Rename to lingua_kit and integrate translation modules"
```

## Step 6: Check Remote Repository

Verify your remote repository is set up:

```powershell
git remote -v
```

If you see output like:
```
origin  https://github.com/yourusername/your-repo.git (fetch)
origin  https://github.com/yourusername/your-repo.git (push)
```

You're good to go! If not, add your remote:

```powershell
git remote add origin https://github.com/yourusername/your-repo-name.git
```

## Step 7: Push to GitHub

### If this is your first push to a new branch:

```powershell
# Push and set upstream
git push -u origin main
```

### If the branch already exists:

```powershell
git push
```

### If you need to force push (use with caution):

```powershell
# Only if you're sure you want to overwrite remote history
git push --force
```

## Step 8: Verify on GitHub

1. Go to your GitHub repository
2. Check that all files are present
3. Verify the `lingua_kit/` directory structure
4. Check that README.md shows the updated information

## Troubleshooting

### Issue: "Large files detected"
If GitHub warns about large files:
- Check `.gitignore` is working
- Remove large files from git history if needed:
  ```powershell
  git rm --cached path/to/large/file
  ```

### Issue: "Remote repository not found"
- Verify the repository URL is correct
- Check you have push permissions
- Create the repository on GitHub first if it doesn't exist

### Issue: "Merge conflicts"
If you have conflicts with remote changes:
```powershell
# Pull remote changes first
git pull origin main

# Resolve conflicts, then:
git add .
git commit -m "Resolve merge conflicts"
git push
```

### Issue: "Authentication failed"
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys for authentication

## Best Practices

1. **Commit frequently** - Make smaller, focused commits
2. **Write clear messages** - Describe what and why, not how
3. **Review before committing** - Use `git status` and `git diff`
4. **Keep .gitignore updated** - Don't commit large files or secrets
5. **Test before pushing** - Make sure code works locally first

## Next Steps After Pushing

1. Create a release tag (optional):
   ```powershell
   git tag -a v1.0.0 -m "Initial release with lingua_kit"
   git push origin v1.0.0
   ```

2. Update repository description on GitHub
3. Add topics/tags to your repository
4. Consider adding a LICENSE file
5. Set up GitHub Actions for CI/CD (optional)

---

**Quick Command Summary:**

```powershell
# Full workflow
git status                          # Check changes
git add .                           # Stage all
git commit -m "Your message"        # Commit
git push origin main                # Push to GitHub
```

