import importlib


def test_project_packages_are_importable():
    package_names = (
        "rater",
        "dataset",
        "fp1_07",
        "fp1_sol01",
        "losses",
        "metric",
        "models",
        "models2",
        "utils",
    )

    for package_name in package_names:
        assert importlib.import_module(package_name).__file__ is not None
