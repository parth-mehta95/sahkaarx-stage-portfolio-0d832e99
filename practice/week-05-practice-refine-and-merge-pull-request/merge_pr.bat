@echo off
echo === 1. Checking out feature branch: feature/data-validation ===
git checkout -B feature/data-validation

echo.
echo === 2. Staging and committing review feedback iterations ===
git add data_processor.py pipeline.py tests/test_data_validation.py pytest.ini README.md practice/week-05-practice-refine-and-merge-pull-request/
git commit -m "refactor(validation): incorporate review feedback with actionable diagnostics, custom rules, and quarantine mode"

echo.
echo === 3. Switching to main branch ===
git checkout main

echo.
echo === 4. Merging feature/data-validation into main with descriptive merge message ===
git merge --no-ff feature/data-validation -m "Merge pull request #1 from feature/data-validation: Incorporate team review feedback, enrich validation diagnostics, and add quarantine error mode"

echo.
echo === 5. Pushing branches to remote origin ===
git push origin feature/data-validation
git push origin main

echo.
echo === Successfully refined and merged PR to main! ===
pause

