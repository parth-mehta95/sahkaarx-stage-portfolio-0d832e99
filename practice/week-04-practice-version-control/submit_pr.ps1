# submit_pr.ps1 - Automated Git Feature Branch & Push Script

Write-Host "=== 1. Checking out branch: feature/data-validation ===" -ForegroundColor Cyan
git checkout -b feature/data-validation

Write-Host "`n=== 2. Staging all files ===" -ForegroundColor Cyan
git add pipeline.py data_processor.py pytest.ini README.md tests/test_data_validation.py practice/week-04-practice-version-control/

Write-Host "`n=== 3. Committing with Conventional Commit message ===" -ForegroundColor Cyan
git commit -m "feat(validation): implement type-hinted data validation functions and pipeline telemetry"

Write-Host "`n=== 4. Pushing branch to origin ===" -ForegroundColor Cyan
git push -u origin feature/data-validation

Write-Host "`n=== Done! Open your Pull Request at the link below: ===" -ForegroundColor Green
Write-Host "https://github.com/parth-mehta95/sahkaarx-stage-portfolio-0d832e99/pull/new/feature/data-validation" -ForegroundColor Yellow

