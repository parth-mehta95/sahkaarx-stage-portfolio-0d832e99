@echo off
echo === 1. Checking out branch: feature/data-validation ===
git checkout -b feature/data-validation

echo.
echo === 2. Staging all files ===
git add pipeline.py data_processor.py pytest.ini README.md tests/test_data_validation.py practice/week-04-practice-version-control/

echo.
echo === 3. Committing with Conventional Commit message ===
git commit -m "feat(validation): implement type-hinted data validation functions and pipeline telemetry"

echo.
echo === 4. Pushing branch to origin ===
git push -u origin feature/data-validation

echo.
echo === Done! Open your Pull Request at the link below: ===
echo https://github.com/parth-mehta95/sahkaarx-stage-portfolio-0d832e99/pull/new/feature/data-validation
pause

