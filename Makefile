# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# Type check
type:
	pytype

# Build the doc
doc:
	cd docs && make html

# Check the spelling in the doc
spelling:
	cd docs && make spelling

# Clean the doc build folder
clean:
	cd docs && make clean
