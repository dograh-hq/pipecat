##############################################################################
# 0. ONE-TIME SETUP
##############################################################################
git clone git@github.com:dograh-hq/pipecat.git
cd pipecat
git remote add upstream https://github.com/pipecat-ai/pipecat.git
git remote -v          # origin = your fork | upstream = original

##############################################################################
# 1. SYNC YOUR MAIN WITH UPSTREAM (NO HISTORY REWRITE)
##############################################################################
git checkout main
git fetch upstream
git merge --ff-only upstream/main   # or: git reset --hard upstream/main
git push origin main                # fork’s main is now 1:1 with upstream

##############################################################################
# 2. START NEW WORK
##############################################################################
git checkout -b feat/<topic>        # always branch off updated main
# ...edit...
git add .
git commit -m "Whatever"
git push -u origin feat/<topic>

##############################################################################
# 3. KEEP YOUR FEATURE BRANCH CURRENT
##############################################################################
# Step A – refresh main
git checkout main
git fetch upstream
git merge --ff-only upstream/main
git push origin main

# Step B – replay your work on top of the new main
git checkout feat/<topic>
git rebase main                     # resolve conflicts once
git push --force-with-lease         # update branch / PR

##############################################################################
# 4. SHIP THE FEATURE
##############################################################################
# Option A – GitHub PR (recommended)
#   base = main, compare = feat/<topic>, squash/rebase-merge, click “Merge”

# Option B – local fast-forward merge
git checkout main
git merge --ff-only feat/<topic>
git push origin main

# Delete branch after merge
git branch -d feat/<topic>
git push origin --delete feat/<topic>

##############################################################################
# 5. HOTFIX DIRECTLY ON MAIN (RARE, SOLO REPO)
##############################################################################
git checkout -b hotfix/issue-123 main
# ...fix...
git commit -m "Hot-fix overflow"
git checkout main
git merge --ff-only hotfix/issue-123
git push origin main

##############################################################################
# 6. INSTALL THE FORK IN ANOTHER PROJECT
##############################################################################
# requirements.txt
pipecat @ git+https://github.com/dograh-hq/pipecat.git@main
# or pin to a tag/sha for reproducible builds
pipecat @ git+https://github.com/dograh-hq/pipecat.git@v0.4.2-fork1
