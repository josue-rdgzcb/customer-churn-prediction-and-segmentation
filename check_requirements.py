# check_requirements.py
# Script to verify installation and versions of libraries listed in requirements.txt

import importlib
import importlib.metadata
import os

# Map Python import names back to pip installation names
IMPORT_MAPPING = {
    "sklearn": "scikit-learn",
    "fontTools": "fonttools",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
    "streamlit_folium": "streamlit-folium"
}

# Map pip installation names to Python import names
LIB_MAPPING = {v: k for k, v in IMPORT_MAPPING.items()}

requirements_file = "requirements.txt"
libraries = []

print("📂 Reading requirements.txt...\n")

if os.path.exists(requirements_file):
    with open(requirements_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Extract package name by removing version operators
            lib_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
            import_name = LIB_MAPPING.get(lib_name.lower(), lib_name)
            
            if import_name not in libraries:
                libraries.append(import_name)
else:
    print(f"❌ Error: '{requirements_file}' not found.")
    exit(1)

print("🔍 Checking installed libraries...\n")

for lib in libraries:
    try:
        module = importlib.import_module(lib)
        pip_name = IMPORT_MAPPING.get(lib, lib)
        
        # Use importlib.metadata to safely retrieve the version from pip
        try:
            version = importlib.metadata.version(pip_name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "Version keyword not found")
            
        print(f"✅ {lib}: {version}")
    except ImportError:
        print(f"❌ {lib}: Not installed")

print("\n🏁 Check complete.")