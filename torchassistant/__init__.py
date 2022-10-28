def _verify_dependency(package_name):
    import importlib

    class DependencyNotInstalledError(Exception):
        pass

    spec = importlib.util.find_spec(package_name)

    if spec is None:
        raise DependencyNotInstalledError(
            f'Cannot find "{package_name}" package. Make sure it is installed'
        )


# confirm that dependencies are installed
_verify_dependency("torch")
_verify_dependency("torchvision")
_verify_dependency("torchmetrics")
