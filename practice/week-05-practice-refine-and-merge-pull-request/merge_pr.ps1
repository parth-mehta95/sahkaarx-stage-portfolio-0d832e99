# merge_pr.ps1 - Incorporate Review Feedback, Commit Iterations, and Merge PR to Main

Write-Host "=== 1. Checking out feature branch: feature/data-validation ===" -ForegroundColor Cyan
git checkout -B feature/data-validation

Write-Host "`n=== 2. Staging and committing review feedback iterations ===" -ForegroundColor Cyan
git add data_processor.py pipeline.py tests/test_data_validation.py pytest.ini README.md practice/week-05-practice-refine-and-merge-pull-request/
git commit -m "refactor(validation): incorporate review feedback with actionable diagnostics, custom rules, and quarantine mode"

Write-Host "`n=== 3. Switching to main branch ===" -ForegroundColor Cyan
git checkout main

Write-Host "`n=== 4. Merging feature/data-validation into main with descriptive merge message ===" -ForegroundColor Cyan
git merge --no-ff feature/data-validation -m "Merge pull request #1 from feature/data-validation: Incorporate team review feedback, enrich validation diagnostics, and add quarantine error mode"

Write-Host "`n=== 5. Pushing branches to remote origin ===" -ForegroundColor Cyan
git push origin feature/data-validation
git push origin main

Write-Host "`n=== Successfully refined and merged PR to main! ===" -ForegroundColor Green

