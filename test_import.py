# Try to import torchcrf with proper error handling
try:
    from TorchCRF import CRF
    print("TorchCRF imported successfully as CRF module")
except ImportError as e:
    print(f"Failed to import TorchCRF as CRF: {e}")
    
try:
    import TorchCRF
    print("TorchCRF imported successfully")
except ImportError as e:
    print(f"Failed to import TorchCRF: {e}")

# Try direct import with full module path
try:
    import importlib
    torchcrf_module = importlib.import_module('TorchCRF')
    print(f"Successfully imported with importlib: {torchcrf_module}")
    print(f"Module file location: {torchcrf_module.__file__ if hasattr(torchcrf_module, '__file__') else 'No file info'}")
    print(f"Module attributes: {[attr for attr in dir(torchcrf_module) if not attr.startswith('_')]}")
except ImportError as e:
    print(f"Failed to import with importlib: {e}")

# List all installed packages to check if it's there
print("\nChecking installed packages...")
import pkg_resources
working_set = pkg_resources.working_set
if working_set is not None:
    installed_packages = [d.project_name for d in working_set]
    torchcrf_packages = [pkg for pkg in installed_packages if 'torch' in pkg.lower() or 'crf' in pkg.lower()]
    print(f"Torch/CRF related packages: {torchcrf_packages}")
else:
    print("Unable to access working_set")

